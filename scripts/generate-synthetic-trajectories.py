#!/usr/bin/env python3
"""Generate synthetic training trajectories by injecting bugs into Rust code
and having a cloud teacher model fix them.

Inspired by Allen AI's SERA approach: define a taxonomy of Rust-specific bug
patterns, inject them into real codebase files, present the buggy code + compiler
error to a teacher LLM, and collect the fix trajectory as SFT training data.

Produces JSONL compatible with unsloth/axolotl and our train-lora.sh pipeline.

Usage:
    python3 scripts/generate-synthetic-trajectories.py
    python3 scripts/generate-synthetic-trajectories.py --num 50 --categories borrow_checker,type_mismatch
    python3 scripts/generate-synthetic-trajectories.py --mode agentic --model gemini-3.1-pro-high
    python3 scripts/generate-synthetic-trajectories.py --dry-run --categories all
    python3 scripts/generate-synthetic-trajectories.py --verify --verify-host root@10.0.0.20

Dependencies: httpx, tqdm (pip install httpx tqdm)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

try:
    import httpx
except ImportError:
    print(
        "Error: httpx is required. Install with: pip install httpx",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from tqdm import tqdm
    from tqdm.asyncio import tqdm as atqdm
except ImportError:
    print(
        "Error: tqdm is required. Install with: pip install tqdm",
        file=sys.stderr,
    )
    sys.exit(1)


# ─── Bug Taxonomy ────────────────────────────────────────────────────────────
# Each pattern aligns with ErrorCategory from coordination/src/feedback/error_parser.rs.
# Categories: borrow_checker, lifetime, type_mismatch, trait_bound, async_await,
#             import_resolution, dead_code, pattern_matching, error_handling,
#             module_visibility

@dataclass
class BugPattern:
    """A single injectable bug pattern with its metadata and code templates."""
    id: str
    category: str
    description: str
    buggy_code: str
    compiler_error: str
    fixed_code: str
    fix_explanation: str
    difficulty: int  # 1=easy, 2=medium, 3=hard
    error_code: str = ""  # rustc error code, e.g. "E0382"
    tags: list[str] = field(default_factory=list)


BUG_PATTERNS: list[BugPattern] = [
    # ── Borrow Checker (complexity 2) ────────────────────────────────────────

    BugPattern(
        id="borrow_double_mut",
        category="borrow_checker",
        description="Two mutable borrows of the same Vec in the same scope",
        buggy_code='''fn process_items(items: &mut Vec<String>) {
    let first = &mut items[0];
    let second = &mut items[1];
    first.push_str(" processed");
    second.push_str(" processed");
}''',
        compiler_error='''error[E0499]: cannot borrow `*items` as mutable more than once at a time
 --> src/lib.rs:3:19
  |
2 |     let first = &mut items[0];
  |                      ----- first mutable borrow occurs here
3 |     let second = &mut items[1];
  |                       ^^^^^ second mutable borrow occurs here
4 |     first.push_str(" processed");
  |     ----- first borrow later used here''',
        fixed_code='''fn process_items(items: &mut Vec<String>) {
    // Split the slice to get non-overlapping mutable references
    let (first_slice, rest) = items.split_at_mut(1);
    first_slice[0].push_str(" processed");
    rest[0].push_str(" processed");
}''',
        fix_explanation="Cannot have two mutable borrows of the same Vec simultaneously. Use split_at_mut() to get non-overlapping mutable slices.",
        difficulty=2,
        error_code="E0499",
        tags=["borrow_checker", "split_at_mut"],
    ),

    BugPattern(
        id="borrow_use_after_move",
        category="borrow_checker",
        description="Use of a String after it has been moved into a function call",
        buggy_code='''fn log_message(msg: String) {
    println!("LOG: {msg}");
}

fn main() {
    let message = String::from("hello world");
    log_message(message);
    println!("Sent: {message}");
}''',
        compiler_error='''error[E0382]: borrow of moved value: `message`
 --> src/main.rs:8:25
  |
6 |     let message = String::from("hello world");
  |         ------- move occurs because `message` has type `String`, which does not implement the `Copy` trait
7 |     log_message(message);
  |                 ------- value moved here
8 |     println!("Sent: {message}");
  |                      ^^^^^^^ value borrowed here after move''',
        fixed_code='''fn log_message(msg: &str) {
    println!("LOG: {msg}");
}

fn main() {
    let message = String::from("hello world");
    log_message(&message);
    println!("Sent: {message}");
}''',
        fix_explanation="The String is moved into log_message, so it can't be used afterward. Change the function to take &str instead, and pass a reference.",
        difficulty=1,
        error_code="E0382",
        tags=["borrow_checker", "move_semantics"],
    ),

    BugPattern(
        id="borrow_mut_and_immut",
        category="borrow_checker",
        description="Simultaneous mutable and immutable borrow of a HashMap",
        buggy_code='''use std::collections::HashMap;

fn update_cache(cache: &mut HashMap<String, Vec<u8>>) {
    for (key, value) in cache.iter() {
        if value.is_empty() {
            cache.remove(key);
        }
    }
}''',
        compiler_error='''error[E0502]: cannot borrow `*cache` as mutable because it is also borrowed as immutable
 --> src/lib.rs:6:13
  |
4 |     for (key, value) in cache.iter() {
  |                          ----------- immutable borrow occurs here
5 |         if value.is_empty() {
6 |             cache.remove(key);
  |             ^^^^^^^^^^^^^^^^^ mutable borrow occurs here
7 |         }
8 |     }
  |     - immutable borrow might be used here, when `cache` is dropped and runs the `Drop` code for type `HashMap`''',
        fixed_code='''use std::collections::HashMap;

fn update_cache(cache: &mut HashMap<String, Vec<u8>>) {
    let keys_to_remove: Vec<String> = cache
        .iter()
        .filter(|(_, value)| value.is_empty())
        .map(|(key, _)| key.clone())
        .collect();
    for key in keys_to_remove {
        cache.remove(&key);
    }
}''',
        fix_explanation="Cannot mutate a HashMap while iterating over it. Collect keys to remove first, then remove them in a separate loop.",
        difficulty=2,
        error_code="E0502",
        tags=["borrow_checker", "iterator_invalidation"],
    ),

    # ── Lifetime Errors (complexity 3) ───────────────────────────────────────

    BugPattern(
        id="lifetime_dangling_ref",
        category="lifetime",
        description="Returning a reference to a local variable",
        buggy_code='''fn longest_line(text: &str) -> &str {
    let lines: Vec<&str> = text.lines().collect();
    let mut longest = "";
    for line in &lines {
        if line.len() > longest.len() {
            longest = line;
        }
    }
    let result = format!("Longest: {longest}");
    &result
}''',
        compiler_error='''error[E0515]: cannot return reference to local variable `result`
 --> src/lib.rs:10:5
  |
10 |     &result
   |     ^^^^^^^ returns a reference to data owned by the current function''',
        fixed_code='''fn longest_line(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let mut longest = "";
    for line in &lines {
        if line.len() > longest.len() {
            longest = line;
        }
    }
    format!("Longest: {longest}")
}''',
        fix_explanation="Cannot return a reference to a local variable that will be dropped. Return an owned String instead.",
        difficulty=2,
        error_code="E0515",
        tags=["lifetime", "owned_vs_borrowed"],
    ),

    BugPattern(
        id="lifetime_struct_ref",
        category="lifetime",
        description="Struct holding a reference without lifetime annotation",
        buggy_code='''struct Config {
    name: &str,
    values: Vec<&str>,
}

impl Config {
    fn new(name: &str) -> Config {
        Config {
            name,
            values: Vec::new(),
        }
    }
}''',
        compiler_error='''error[E0106]: missing lifetime specifier
 --> src/lib.rs:2:11
  |
2 |     name: &str,
  |           ^ expected named lifetime parameter

error[E0106]: missing lifetime specifier
 --> src/lib.rs:3:16
  |
3 |     values: Vec<&str>,
  |                 ^ expected named lifetime parameter''',
        fixed_code='''struct Config<'a> {
    name: &'a str,
    values: Vec<&'a str>,
}

impl<'a> Config<'a> {
    fn new(name: &'a str) -> Config<'a> {
        Config {
            name,
            values: Vec::new(),
        }
    }
}''',
        fix_explanation="References in structs require explicit lifetime annotations. Add a lifetime parameter 'a to the struct and annotate all references.",
        difficulty=3,
        error_code="E0106",
        tags=["lifetime", "struct_lifetime"],
    ),

    BugPattern(
        id="lifetime_closure_capture",
        category="lifetime",
        description="Closure captures a reference that does not live long enough",
        buggy_code='''fn make_greeting(name: &str) -> Box<dyn Fn() -> String> {
    Box::new(move || format!("Hello, {name}!"))
}''',
        compiler_error='''error[E0621]: explicit lifetime required in the type of `name`
 --> src/lib.rs:2:5
  |
1 | fn make_greeting(name: &str) -> Box<dyn Fn() -> String> {
  |                        ---- help: add explicit lifetime `'static` to the type of `name`: `&'static str`
2 |     Box::new(move || format!("Hello, {name}!"))
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ lifetime `'static` required''',
        fixed_code='''fn make_greeting(name: &str) -> Box<dyn Fn() -> String> {
    let name = name.to_owned();
    Box::new(move || format!("Hello, {name}!"))
}''',
        fix_explanation="The closure outlives the borrowed &str. Clone the string into an owned String before capturing it in the closure.",
        difficulty=2,
        error_code="E0621",
        tags=["lifetime", "closure", "owned_vs_borrowed"],
    ),

    # ── Type Mismatch (complexity 1) ─────────────────────────────────────────

    BugPattern(
        id="type_option_unwrap",
        category="type_mismatch",
        description="Using Option<T> where T is expected without unwrapping",
        buggy_code='''fn find_user(id: u64) -> Option<String> {
    if id == 1 {
        Some("alice".to_string())
    } else {
        None
    }
}

fn greet_user(id: u64) -> String {
    let name = find_user(id);
    format!("Hello, {name}!")
}''',
        compiler_error='''error[E0308]: mismatched types
  --> src/lib.rs:11:27
   |
11 |     format!("Hello, {name}!")
   |                      ^^^^ expected `String`, found `Option<String>`
   |
   = note: expected struct `String`
              found enum `Option<String>`''',
        fixed_code='''fn find_user(id: u64) -> Option<String> {
    if id == 1 {
        Some("alice".to_string())
    } else {
        None
    }
}

fn greet_user(id: u64) -> String {
    let name = find_user(id).unwrap_or_else(|| "stranger".to_string());
    format!("Hello, {name}!")
}''',
        fix_explanation="find_user returns Option<String> but format! expects String. Use unwrap_or_else to provide a fallback value.",
        difficulty=1,
        error_code="E0308",
        tags=["type_mismatch", "option"],
    ),

    BugPattern(
        id="type_integer_mismatch",
        category="type_mismatch",
        description="Mixing u32 and usize in array indexing",
        buggy_code='''fn get_nth_element(data: &[String], index: u32) -> &str {
    &data[index]
}''',
        compiler_error='''error[E0277]: the type `[String]` cannot be indexed by `u32`
 --> src/lib.rs:2:10
  |
2 |     &data[index]
  |          ^^^^^^^ slice indices are of type `usize` or ranges of `usize`
  |
  = help: the trait `SliceIndex<[String]>` is not implemented for `u32`''',
        fixed_code='''fn get_nth_element(data: &[String], index: u32) -> &str {
    &data[index as usize]
}''',
        fix_explanation="Rust slices are indexed by usize, not u32. Cast the index with 'as usize'.",
        difficulty=1,
        error_code="E0277",
        tags=["type_mismatch", "integer_cast"],
    ),

    BugPattern(
        id="type_string_vs_str",
        category="type_mismatch",
        description="Passing String where &str is expected and vice versa",
        buggy_code='''fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().to_string() + chars.as_str(),
    }
}

fn process_names(names: Vec<String>) -> Vec<String> {
    names.iter().map(capitalize).collect()
}''',
        compiler_error='''error[E0631]: type mismatch in function arguments
  --> src/lib.rs:10:25
   |
1  | fn capitalize(s: &str) -> String {
   | -------------------------------- found signature defined here
...
10 |     names.iter().map(capitalize).collect()
   |                 --- ^^^^^^^^^^ expected due to this
   |                 |
   |                 required by a bound introduced by this call
   |
   = note: expected function signature `fn(&String) -> _`
              found function signature `fn(&str) -> _`''',
        fixed_code='''fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().to_string() + chars.as_str(),
    }
}

fn process_names(names: Vec<String>) -> Vec<String> {
    names.iter().map(|s| capitalize(s)).collect()
}''',
        fix_explanation="iter() yields &String, but capitalize expects &str. Use a closure to auto-deref, or call .map(|s| capitalize(s)) which coerces &String to &str.",
        difficulty=1,
        error_code="E0631",
        tags=["type_mismatch", "string_str_coercion"],
    ),

    # ── Trait Bound Violations (complexity 2) ────────────────────────────────

    BugPattern(
        id="trait_missing_derive",
        category="trait_bound",
        description="Using Debug/Clone on a struct that doesn't derive them",
        buggy_code='''struct TaskResult {
    status: String,
    output: Vec<u8>,
    duration_ms: u64,
}

fn log_result(result: &TaskResult) {
    println!("Result: {:?}", result);
}

fn duplicate_result(result: &TaskResult) -> TaskResult {
    result.clone()
}''',
        compiler_error='''error[E0277]: `TaskResult` doesn't implement `Debug`
 --> src/lib.rs:8:34
  |
8 |     println!("Result: {:?}", result);
  |                              ^^^^^^ `TaskResult` cannot be formatted using `{:?}`
  |
  = help: the trait `Debug` is not implemented for `TaskResult`
  = note: add `#[derive(Debug)]` to `TaskResult` or manually `impl Debug for TaskResult`

error[E0599]: no method named `clone` found for reference `&TaskResult` in the current scope
  --> src/lib.rs:12:12
   |
12 |     result.clone()
   |            ^^^^^ method not found in `&TaskResult`
   |
   = help: items from traits can only be used if the trait is implemented and in scope
   = note: the following trait defines an item `clone`, perhaps you need to implement it:
           candidate #1: `Clone`''',
        fixed_code='''#[derive(Debug, Clone)]
struct TaskResult {
    status: String,
    output: Vec<u8>,
    duration_ms: u64,
}

fn log_result(result: &TaskResult) {
    println!("Result: {:?}", result);
}

fn duplicate_result(result: &TaskResult) -> TaskResult {
    result.clone()
}''',
        fix_explanation="Add #[derive(Debug, Clone)] to the struct. All fields (String, Vec<u8>, u64) already implement both traits.",
        difficulty=1,
        error_code="E0277",
        tags=["trait_bound", "derive"],
    ),

    BugPattern(
        id="trait_send_bound",
        category="trait_bound",
        description="Spawning a tokio task with a non-Send type",
        buggy_code='''use std::rc::Rc;

async fn process_batch(items: Vec<String>) {
    let shared = Rc::new(items);
    tokio::spawn(async move {
        for item in shared.iter() {
            println!("{item}");
        }
    });
}''',
        compiler_error='''error: future cannot be sent between threads safely
 --> src/lib.rs:5:5
  |
5 |     tokio::spawn(async move {
  |     ^^^^^^^^^^^^ future created by async block is not `Send`
  |
  = help: within `impl Future<Output = ()>`, the trait `Send` is not implemented for `Rc<Vec<String>>`
  = note: required for `Rc<Vec<String>>` to implement `Send`
note: future is not `Send` as this value is used across an await
  = note: required by a bound in `tokio::spawn`''',
        fixed_code='''use std::sync::Arc;

async fn process_batch(items: Vec<String>) {
    let shared = Arc::new(items);
    tokio::spawn(async move {
        for item in shared.iter() {
            println!("{item}");
        }
    });
}''',
        fix_explanation="Rc is not Send, so it can't cross thread boundaries. Replace Rc with Arc (atomic reference counting) for thread-safe sharing.",
        difficulty=2,
        error_code="E0277",
        tags=["trait_bound", "send_sync", "async"],
    ),

    BugPattern(
        id="trait_from_impl",
        category="trait_bound",
        description="Using .into() without a From impl for the target type",
        buggy_code='''struct AppError {
    message: String,
    code: u32,
}

fn read_config(path: &str) -> Result<String, AppError> {
    std::fs::read_to_string(path).map_err(|e| e.into())
}''',
        compiler_error='''error[E0277]: the trait bound `AppError: From<std::io::Error>` is not satisfied
 --> src/lib.rs:7:49
  |
7 |     std::fs::read_to_string(path).map_err(|e| e.into())
  |                                                 ^^^^ the trait `From<std::io::Error>` is not implemented for `AppError`''',
        fixed_code='''struct AppError {
    message: String,
    code: u32,
}

impl From<std::io::Error> for AppError {
    fn from(err: std::io::Error) -> Self {
        AppError {
            message: err.to_string(),
            code: 1,
        }
    }
}

fn read_config(path: &str) -> Result<String, AppError> {
    std::fs::read_to_string(path).map_err(|e| e.into())
}''',
        fix_explanation="Implement From<std::io::Error> for AppError so the .into() conversion works.",
        difficulty=2,
        error_code="E0277",
        tags=["trait_bound", "from_into", "error_handling"],
    ),

    # ── Async/Await Issues (complexity 3) ────────────────────────────────────

    BugPattern(
        id="async_missing_await",
        category="async_await",
        description="Calling an async function without .await",
        buggy_code='''async fn fetch_data(url: &str) -> Result<String, reqwest::Error> {
    reqwest::get(url).text()
}''',
        compiler_error='''error[E0599]: no method named `text` found for opaque type `impl Future<Output = Result<Response, Error>>` in the current scope
 --> src/lib.rs:2:23
  |
2 |     reqwest::get(url).text()
  |                       ^^^^ method not found in `impl Future<Output = Result<Response, Error>>`
  |
help: consider `await`ing on the `Future` and calling the method on its `Output`
  |
2 |     reqwest::get(url).await.text()
  |                       ++++++''',
        fixed_code='''async fn fetch_data(url: &str) -> Result<String, reqwest::Error> {
    reqwest::get(url).await?.text().await
}''',
        fix_explanation="reqwest::get() returns a Future, not a Response. Add .await to get the Response, then .await again for .text().",
        difficulty=1,
        error_code="E0599",
        tags=["async_await", "missing_await"],
    ),

    BugPattern(
        id="async_not_send",
        category="async_await",
        description="Holding a MutexGuard across an .await point",
        buggy_code='''use std::sync::Mutex;

struct SharedState {
    counter: Mutex<u64>,
}

impl SharedState {
    async fn increment_and_log(&self) {
        let mut guard = self.counter.lock().unwrap();
        *guard += 1;
        self.log_value(*guard).await;
    }

    async fn log_value(&self, val: u64) {
        println!("Counter: {val}");
    }
}''',
        compiler_error='''error: future cannot be sent between threads safely
  --> src/lib.rs:9:5
   |
9  |     async fn increment_and_log(&self) {
   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ future created by async block is not `Send`
   |
   = help: within the async body, the type `MutexGuard<'_, u64>` is not `Send`
   = note: required for the cast to the object type `dyn Future<Output = ()> + Send`
note: future is not `Send` as this value is used across an await
  --> src/lib.rs:12:9
   |
10 |         let mut guard = self.counter.lock().unwrap();
   |             --------- has type `MutexGuard<'_, u64>` which is not `Send`
11 |         *guard += 1;
12 |         self.log_value(*guard).await;
   |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ await occurs here, with `guard` maybe used later''',
        fixed_code='''use std::sync::Mutex;

struct SharedState {
    counter: Mutex<u64>,
}

impl SharedState {
    async fn increment_and_log(&self) {
        let val = {
            let mut guard = self.counter.lock().unwrap();
            *guard += 1;
            *guard
        };
        self.log_value(val).await;
    }

    async fn log_value(&self, val: u64) {
        println!("Counter: {val}");
    }
}''',
        fix_explanation="MutexGuard is not Send. Drop the guard before the .await point by scoping the lock in a block and extracting the value.",
        difficulty=3,
        error_code="E0277",
        tags=["async_await", "send_sync", "mutex_guard"],
    ),

    BugPattern(
        id="async_lifetime_in_trait",
        category="async_await",
        description="Async trait method returning a reference needs lifetime bounds",
        buggy_code='''trait DataStore {
    async fn get(&self, key: &str) -> Option<&str>;
}

struct MemStore {
    data: std::collections::HashMap<String, String>,
}

impl DataStore for MemStore {
    async fn get(&self, key: &str) -> Option<&str> {
        self.data.get(key).map(|s| s.as_str())
    }
}''',
        compiler_error='''error[E0726]: implicit elided lifetime not allowed here
  --> src/lib.rs:2:48
   |
2  |     async fn get(&self, key: &str) -> Option<&str>;
   |                                                ^^^^ expected lifetime parameter
   |
   = note: async fn's return type must be fully specified in traits''',
        fixed_code='''trait DataStore {
    fn get(&self, key: &str) -> impl std::future::Future<Output = Option<&str>> + Send + '_;
}

struct MemStore {
    data: std::collections::HashMap<String, String>,
}

impl DataStore for MemStore {
    fn get(&self, key: &str) -> impl std::future::Future<Output = Option<&str>> + Send + '_ {
        async move {
            self.data.get(key).map(|s| s.as_str())
        }
    }
}''',
        fix_explanation="Async trait methods with references in the return type need explicit lifetime handling. Use RPITIT (return-position impl Trait in Trait) with '_ lifetime bound.",
        difficulty=3,
        error_code="E0726",
        tags=["async_await", "lifetime", "trait"],
    ),

    # ── Import Resolution (complexity 1) ─────────────────────────────────────

    BugPattern(
        id="import_missing_use",
        category="import_resolution",
        description="Using HashMap without importing it",
        buggy_code='''fn count_words(text: &str) -> HashMap<&str, usize> {
    let mut counts = HashMap::new();
    for word in text.split_whitespace() {
        *counts.entry(word).or_insert(0) += 1;
    }
    counts
}''',
        compiler_error='''error[E0433]: failed to resolve: use of undeclared type `HashMap`
 --> src/lib.rs:1:31
  |
1 | fn count_words(text: &str) -> HashMap<&str, usize> {
  |                               ^^^^^^^ not found in this scope
  |
help: consider importing this struct
  |
1 + use std::collections::HashMap;
  |''',
        fixed_code='''use std::collections::HashMap;

fn count_words(text: &str) -> HashMap<&str, usize> {
    let mut counts = HashMap::new();
    for word in text.split_whitespace() {
        *counts.entry(word).or_insert(0) += 1;
    }
    counts
}''',
        fix_explanation="HashMap is in std::collections, not the prelude. Add 'use std::collections::HashMap;'.",
        difficulty=1,
        error_code="E0433",
        tags=["import_resolution"],
    ),

    BugPattern(
        id="import_trait_method",
        category="import_resolution",
        description="Calling a trait method without importing the trait",
        buggy_code='''use std::fs::File;

fn open_buffered(path: &str) -> std::io::Result<std::io::BufReader<File>> {
    let file = File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    Ok(reader)
}''',
        compiler_error='''error[E0599]: no method named `read_line` found for struct `BufReader<File>` in the current scope
 --> src/lib.rs:7:12
  |
7 |     reader.read_line(&mut line)?;
  |            ^^^^^^^^^ method not found in `BufReader<File>`
  |
  = help: items from traits can only be used if the trait is implemented and in scope
help: the following trait is implemented but not in scope; perhaps add a `use` for it:
  |
1 + use std::io::BufRead;
  |''',
        fixed_code='''use std::fs::File;
use std::io::BufRead;

fn open_buffered(path: &str) -> std::io::Result<std::io::BufReader<File>> {
    let file = File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    Ok(reader)
}''',
        fix_explanation="read_line is a method on the BufRead trait, which must be in scope. Add 'use std::io::BufRead;'.",
        difficulty=1,
        error_code="E0599",
        tags=["import_resolution", "trait_import"],
    ),

    # ── Dead Code / Unused Variables (complexity 1) ──────────────────────────

    BugPattern(
        id="dead_code_unused_result",
        category="dead_code",
        description="Ignoring a Result that should be handled (must_use)",
        buggy_code='''use std::fs;

fn cleanup_temp_files(dir: &str) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|ext| ext == "tmp") {
            fs::remove_file(&path);
        }
    }
}''',
        compiler_error='''warning: unused `Result` that must be used
 --> src/lib.rs:7:13
  |
7 |             fs::remove_file(&path);
  |             ^^^^^^^^^^^^^^^^^^^^^^
  |
  = note: this `Result` may be an `Err` variant, which should be handled
  = note: `#[warn(unused_must_use)]` on by default
help: use `let _ = ...` to ignore the resulting value
  |
7 |             let _ = fs::remove_file(&path);
  |             +++++++''',
        fixed_code='''use std::fs;

fn cleanup_temp_files(dir: &str) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|ext| ext == "tmp") {
            if let Err(e) = fs::remove_file(&path) {
                eprintln!("Warning: failed to remove {}: {e}", path.display());
            }
        }
    }
}''',
        fix_explanation="fs::remove_file returns a Result that must be handled. Use if-let-Err to log failures gracefully.",
        difficulty=1,
        error_code="",
        tags=["dead_code", "must_use", "error_handling"],
    ),

    BugPattern(
        id="dead_code_unused_variable",
        category="dead_code",
        description="Variable assigned but never read",
        buggy_code='''fn parse_header(raw: &[u8]) -> Option<(String, String)> {
    let header_str = std::str::from_utf8(raw).ok()?;
    let parts: Vec<&str> = header_str.splitn(2, ':').collect();
    let key = parts.first()?.trim().to_string();
    let value = parts.get(1)?.trim().to_string();
    let normalized_key = key.to_lowercase();
    Some((key, value))
}''',
        compiler_error='''warning: unused variable: `normalized_key`
 --> src/lib.rs:6:9
  |
6 |     let normalized_key = key.to_lowercase();
  |         ^^^^^^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_normalized_key`
  |
  = note: `#[warn(unused_variables)]` on by default''',
        fixed_code='''fn parse_header(raw: &[u8]) -> Option<(String, String)> {
    let header_str = std::str::from_utf8(raw).ok()?;
    let parts: Vec<&str> = header_str.splitn(2, ':').collect();
    let key = parts.first()?.trim().to_lowercase();
    let value = parts.get(1)?.trim().to_string();
    Some((key, value))
}''',
        fix_explanation="normalized_key is computed but never used. The intent was likely to use the lowercase version as the key. Apply to_lowercase() directly.",
        difficulty=1,
        error_code="",
        tags=["dead_code", "unused_variable"],
    ),

    # ── Pattern Matching Exhaustiveness (complexity 2) ───────────────────────

    BugPattern(
        id="pattern_non_exhaustive",
        category="pattern_matching",
        description="Non-exhaustive match on an enum",
        buggy_code='''enum Status {
    Pending,
    Running,
    Success,
    Failed(String),
    Cancelled,
}

fn status_icon(status: &Status) -> &str {
    match status {
        Status::Pending => "⏳",
        Status::Running => "🔄",
        Status::Success => "✅",
        Status::Failed(_) => "❌",
    }
}''',
        compiler_error='''error[E0004]: non-exhaustive patterns: `&Status::Cancelled` not covered
  --> src/lib.rs:10:11
   |
10 |     match status {
   |           ^^^^^^ pattern `&Status::Cancelled` not covered
   |
note: `Status` defined here
  --> src/lib.rs:6:5
   |
1  | enum Status {
   |      ------
...
6  |     Cancelled,
   |     ^^^^^^^^^ not covered
   = note: the matched value is of type `&Status`
help: ensure that all possible cases are being handled by adding a match arm with a wildcard pattern or an explicit pattern as shown
   |
14 ~         Status::Failed(_) => "❌",
15 +         &Status::Cancelled => todo!(),
   |''',
        fixed_code='''enum Status {
    Pending,
    Running,
    Success,
    Failed(String),
    Cancelled,
}

fn status_icon(status: &Status) -> &str {
    match status {
        Status::Pending => "⏳",
        Status::Running => "🔄",
        Status::Success => "✅",
        Status::Failed(_) => "❌",
        Status::Cancelled => "🚫",
    }
}''',
        fix_explanation="The match is missing the Cancelled variant. Add an arm for it.",
        difficulty=1,
        error_code="E0004",
        tags=["pattern_matching", "exhaustiveness"],
    ),

    BugPattern(
        id="pattern_refutable_let",
        category="pattern_matching",
        description="Using a refutable pattern in a let binding",
        buggy_code='''fn first_two(data: &[u8]) -> (u8, u8) {
    let [a, b, ..] = data;
    (*a, *b)
}''',
        compiler_error='''error[E0005]: refutable pattern in local binding
 --> src/lib.rs:2:9
  |
2 |     let [a, b, ..] = data;
  |         ^^^^^^^^^^ pattern `&[]` and `&[_]` not covered
  |
  = note: `let` bindings require an "irrefutable pattern", like a `struct` or an `enum` with only one variant
  = note: for more information, visit https://doc.rust-lang.org/book/ch18-02-refutability.html
help: you might want to use `let else` to handle the refuted case
  |
2 |     let [a, b, ..] = data else { todo!() };
  |                            ++++++++++++++++''',
        fixed_code='''fn first_two(data: &[u8]) -> Option<(u8, u8)> {
    if let [a, b, ..] = data {
        Some((*a, *b))
    } else {
        None
    }
}''',
        fix_explanation="The slice pattern [a, b, ..] is refutable — the slice might have fewer than 2 elements. Use if-let or return Option to handle the failure case.",
        difficulty=2,
        error_code="E0005",
        tags=["pattern_matching", "refutable_pattern"],
    ),

    # ── Error Handling (complexity 1-2) ──────────────────────────────────────

    BugPattern(
        id="error_unwrap_none",
        category="error_handling",
        description="Calling unwrap() on a None value from HashMap::get",
        buggy_code='''use std::collections::HashMap;

fn get_setting(config: &HashMap<String, String>, key: &str) -> String {
    config.get(key).unwrap().clone()
}''',
        compiler_error='''thread 'main' panicked at 'called `Option::unwrap()` on a `None` value', src/lib.rs:4:21''',
        fixed_code='''use std::collections::HashMap;

fn get_setting(config: &HashMap<String, String>, key: &str) -> Option<String> {
    config.get(key).cloned()
}''',
        fix_explanation="HashMap::get returns Option, which may be None. Return Option<String> instead of panicking with unwrap().",
        difficulty=1,
        error_code="",
        tags=["error_handling", "unwrap", "option"],
    ),

    BugPattern(
        id="error_question_mark_mismatch",
        category="error_handling",
        description="Using ? operator with incompatible error types",
        buggy_code='''use std::num::ParseIntError;

fn parse_config_value(raw: &str) -> Result<u64, String> {
    let trimmed = raw.trim();
    let value: u64 = trimmed.parse()?;
    Ok(value)
}''',
        compiler_error='''error[E0277]: `?` couldn't convert the error to `String`
 --> src/lib.rs:5:38
  |
3 | fn parse_config_value(raw: &str) -> Result<u64, String> {
  |                                     ------------------- expected `String` because of this
4 |     let trimmed = raw.trim();
5 |     let value: u64 = trimmed.parse()?;
  |                                      ^ the trait `From<ParseIntError>` is not implemented for `String`
  |
  = note: the question mark operation (`?`) implicitly performs a conversion on the error value using the `From` trait
  = note: required for `Result<u64, String>` to implement `FromResidual<Result<Infallible, ParseIntError>>`''',
        fixed_code='''fn parse_config_value(raw: &str) -> Result<u64, String> {
    let trimmed = raw.trim();
    let value: u64 = trimmed.parse().map_err(|e: std::num::ParseIntError| e.to_string())?;
    Ok(value)
}''',
        fix_explanation="The ? operator needs From<ParseIntError> for String, which doesn't exist. Use .map_err() to convert the error explicitly.",
        difficulty=2,
        error_code="E0277",
        tags=["error_handling", "question_mark", "from_trait"],
    ),

    # ── Module Visibility (complexity 1-2) ───────────────────────────────────

    BugPattern(
        id="visibility_private_field",
        category="module_visibility",
        description="Accessing a private struct field from outside the module",
        buggy_code='''mod config {
    pub struct Settings {
        host: String,
        port: u16,
    }

    impl Settings {
        pub fn new(host: String, port: u16) -> Self {
            Settings { host, port }
        }
    }
}

fn main() {
    let settings = config::Settings::new("localhost".into(), 8080);
    println!("Connecting to {}:{}", settings.host, settings.port);
}''',
        compiler_error='''error[E0616]: field `host` of struct `Settings` is private
  --> src/main.rs:16:48
   |
16 |     println!("Connecting to {}:{}", settings.host, settings.port);
   |                                              ^^^^ private field

error[E0616]: field `port` of struct `Settings` is private
  --> src/main.rs:16:63
   |
16 |     println!("Connecting to {}:{}", settings.host, settings.port);
   |                                                             ^^^^ private field''',
        fixed_code='''mod config {
    pub struct Settings {
        host: String,
        port: u16,
    }

    impl Settings {
        pub fn new(host: String, port: u16) -> Self {
            Settings { host, port }
        }

        pub fn host(&self) -> &str {
            &self.host
        }

        pub fn port(&self) -> u16 {
            self.port
        }
    }
}

fn main() {
    let settings = config::Settings::new("localhost".into(), 8080);
    println!("Connecting to {}:{}", settings.host(), settings.port());
}''',
        fix_explanation="Struct fields are private by default. Add public getter methods instead of making fields pub (better encapsulation).",
        difficulty=1,
        error_code="E0616",
        tags=["module_visibility", "encapsulation"],
    ),

    BugPattern(
        id="visibility_pub_crate",
        category="module_visibility",
        description="Accessing a pub(crate) item from a downstream crate",
        buggy_code='''// In crate `coordination`:
pub(crate) fn internal_helper() -> String {
    "helper result".to_string()
}

// In crate `swarm-agents`:
fn use_helper() -> String {
    coordination::internal_helper()
}''',
        compiler_error='''error[E0603]: function `internal_helper` is private
 --> crates/swarm-agents/src/lib.rs:3:19
  |
3 |     coordination::internal_helper()
  |                   ^^^^^^^^^^^^^^^ private function
  |
note: the function `internal_helper` is defined here
 --> coordination/src/lib.rs:2:1
  |
2 | pub(crate) fn internal_helper() -> String {
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^''',
        fixed_code='''// In crate `coordination`:
pub fn internal_helper() -> String {
    "helper result".to_string()
}

// In crate `swarm-agents`:
fn use_helper() -> String {
    coordination::internal_helper()
}''',
        fix_explanation="pub(crate) makes the function visible only within the same crate. Change to pub to make it accessible from dependent crates.",
        difficulty=1,
        error_code="E0603",
        tags=["module_visibility", "pub_crate"],
    ),
]


# Map category names to their IDs for filtering
CATEGORIES = sorted({p.category for p in BUG_PATTERNS})


# ─── System Prompts ──────────────────────────────────────────────────────────

SYSTEM_PROMPT_DIRECT = """\
You are an expert Rust developer. You will be given Rust code that contains a \
compilation error, along with the compiler error message. Your task is to:

1. Analyze the error and explain the root cause concisely.
2. Provide the corrected code that fixes the error.

Your response must include:
- A brief explanation of what went wrong (1-3 sentences).
- The complete fixed code in a ```rust code block.

Do not add features, refactor unnecessarily, or change behavior beyond fixing \
the compilation error. Minimize the diff."""

SYSTEM_PROMPT_AGENTIC = """\
You are a Rust coding agent working in an autonomous swarm. You have access to \
the following tools:

- read_file(path): Read a source file from the repository
- edit_file(path, old_text, new_text): Replace exact text in a file
- run_command(command): Run a shell command (cargo check, cargo test, etc.)

Your task is to fix the compilation error described in the issue. Follow this \
workflow:

1. Read the relevant file to understand the full context.
2. Analyze the error and identify the root cause.
3. Apply the minimal fix using edit_file.
4. Verify the fix compiles with run_command("cargo check").

Be precise with edit_file — the old_text must match exactly. Keep changes minimal."""


# ─── Trajectory Generation ──────────────────────────────────────────────────

@dataclass
class TrajectoryResult:
    """Result of a single trajectory generation attempt."""
    success: bool
    messages: list[dict[str, str]] | None = None
    metadata: dict[str, Any] | None = None
    error: str | None = None


def build_direct_prompt(pattern: BugPattern, source_file: str | None = None) -> list[dict[str, str]]:
    """Build a direct (single-turn) prompt for the teacher model."""
    user_content = (
        f"The following Rust code has a compilation error:\n\n"
        f"```rust\n{pattern.buggy_code}\n```\n\n"
        f"Compiler error:\n```\n{pattern.compiler_error}\n```\n\n"
        f"Fix the code."
    )
    if source_file:
        user_content = f"File: {source_file}\n\n" + user_content

    return [
        {"role": "system", "content": SYSTEM_PROMPT_DIRECT},
        {"role": "user", "content": user_content},
    ]


def build_agentic_prompt(
    pattern: BugPattern,
    source_file: str = "src/lib.rs",
) -> list[dict[str, str]]:
    """Build a multi-turn agentic trajectory with tool calls.

    This simulates the swarm worker workflow: read → analyze → edit → verify.
    We construct the full conversation including synthetic tool results.
    """
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_AGENTIC,
        },
        {
            "role": "user",
            "content": (
                f"cargo check fails with the following error in {source_file}:\n\n"
                f"```\n{pattern.compiler_error}\n```\n\n"
                f"Fix this compilation error."
            ),
        },
        {
            "role": "assistant",
            "content": (
                f"Let me read the file to understand the full context.\n\n"
                f'<tool_call>read_file("{source_file}")</tool_call>'
            ),
        },
        {
            "role": "user",
            "content": f"<tool_result>\n{pattern.buggy_code}\n</tool_result>",
        },
    ]
    # The assistant analysis + edit will be generated by the teacher model.
    # We provide the context up to this point and let the model complete.
    return messages


def build_agentic_gold_response(pattern: BugPattern, source_file: str = "src/lib.rs") -> str:
    """Build the gold-standard agentic response for when we skip the API call."""
    # Compute the minimal old/new text for the edit_file call.
    # For simplicity, use the full buggy/fixed code as the search/replace.
    return (
        f"I see the issue: {pattern.fix_explanation}\n\n"
        f'<tool_call>edit_file("{source_file}", '
        f'"{_escape_for_tool(pattern.buggy_code)}", '
        f'"{_escape_for_tool(pattern.fixed_code)}")</tool_call>'
    )


def build_agentic_verification_turns(pattern: BugPattern, source_file: str = "src/lib.rs") -> list[dict[str, str]]:
    """Build the verification turns after the edit for agentic trajectories."""
    return [
        {
            "role": "user",
            "content": "<tool_result>File edited successfully.</tool_result>",
        },
        {
            "role": "assistant",
            "content": 'Now let me verify the fix compiles.\n\n<tool_call>run_command("cargo check")</tool_call>',
        },
        {
            "role": "user",
            "content": "<tool_result>    Compiling myproject v0.1.0\n    Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.23s</tool_result>",
        },
        {
            "role": "assistant",
            "content": (
                f"The fix compiles successfully. Here's what was wrong and how I fixed it:\n\n"
                f"**Root cause:** {pattern.fix_explanation}\n\n"
                f"**Fix applied in** `{source_file}` — the corrected code:\n\n"
                f"```rust\n{pattern.fixed_code}\n```"
            ),
        },
    ]


def _escape_for_tool(code: str) -> str:
    """Escape a code string for embedding in a tool_call representation."""
    return code.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


async def call_teacher_api(
    client: httpx.AsyncClient,
    messages: list[dict[str, str]],
    model: str,
    api_url: str,
    api_key: str,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    retries: int = 3,
) -> str | None:
    """Call the teacher model via OpenAI-compatible API with retry logic.

    Returns the assistant content string, or None on failure.
    """
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["x-api-key"] = api_key  # CLIAPIProxy uses this header

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_error = None
    for attempt in range(retries):
        try:
            resp = await client.post(
                f"{api_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=120.0,
            )
            if resp.status_code == 429:
                # Rate limited — back off
                wait = min(2 ** attempt * 5, 60)
                print(f"  Rate limited, waiting {wait}s...", file=sys.stderr)
                await asyncio.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                last_error = "Empty choices in API response"
                continue

            content = choices[0].get("message", {}).get("content", "")
            if content:
                return content
            last_error = "Empty content in API response"

        except httpx.TimeoutException:
            last_error = f"Request timed out (attempt {attempt + 1}/{retries})"
            print(f"  {last_error}", file=sys.stderr)
        except httpx.HTTPStatusError as e:
            last_error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            print(f"  API error: {last_error}", file=sys.stderr)
        except Exception as e:
            last_error = f"Unexpected error: {e}"
            print(f"  {last_error}", file=sys.stderr)

        if attempt < retries - 1:
            wait = 2 ** attempt * 2
            await asyncio.sleep(wait)

    return None


async def generate_direct_trajectory(
    client: httpx.AsyncClient,
    pattern: BugPattern,
    model: str,
    api_url: str,
    api_key: str,
    source_file: str | None = None,
    dry_run: bool = False,
) -> TrajectoryResult:
    """Generate a single direct (single-turn Q&A) trajectory."""
    prompt_messages = build_direct_prompt(pattern, source_file)

    if dry_run:
        return TrajectoryResult(
            success=True,
            messages=prompt_messages + [{"role": "assistant", "content": "[DRY RUN — would call teacher API]"}],
            metadata=_build_metadata(pattern, model, source_file, verified=False),
        )

    response = await call_teacher_api(
        client, prompt_messages, model, api_url, api_key,
    )
    if response is None:
        return TrajectoryResult(success=False, error="Teacher API call failed after retries")

    messages = prompt_messages + [{"role": "assistant", "content": response}]

    return TrajectoryResult(
        success=True,
        messages=messages,
        metadata=_build_metadata(pattern, model, source_file, verified=False),
    )


async def generate_agentic_trajectory(
    client: httpx.AsyncClient,
    pattern: BugPattern,
    model: str,
    api_url: str,
    api_key: str,
    source_file: str = "src/lib.rs",
    dry_run: bool = False,
) -> TrajectoryResult:
    """Generate a multi-turn agentic trajectory with tool calls.

    The conversation flow:
    1. System prompt with tool descriptions
    2. User presents the error
    3. Assistant reads the file (tool call)
    4. User provides file content (tool result)
    5. Assistant analyzes and applies fix (generated by teacher)
    6. User confirms edit success (tool result)
    7. Assistant verifies with cargo check (tool call)
    8. User provides check output (tool result)
    9. Assistant summarizes
    """
    prompt_messages = build_agentic_prompt(pattern, source_file)

    if dry_run:
        gold = build_agentic_gold_response(pattern, source_file)
        verification = build_agentic_verification_turns(pattern, source_file)
        all_messages = prompt_messages + [{"role": "assistant", "content": gold}] + verification
        return TrajectoryResult(
            success=True,
            messages=all_messages,
            metadata=_build_metadata(pattern, model, source_file, verified=False, mode="agentic"),
        )

    # Call teacher to get the analysis + edit step
    response = await call_teacher_api(
        client, prompt_messages, model, api_url, api_key,
        temperature=0.3,
        max_tokens=3072,
    )
    if response is None:
        return TrajectoryResult(success=False, error="Teacher API call failed for agentic trajectory")

    # Build the complete conversation
    all_messages = prompt_messages + [{"role": "assistant", "content": response}]

    # Add verification turns with synthetic tool results
    all_messages.extend(build_agentic_verification_turns(pattern, source_file))

    return TrajectoryResult(
        success=True,
        messages=all_messages,
        metadata=_build_metadata(pattern, model, source_file, verified=False, mode="agentic"),
    )


def _build_metadata(
    pattern: BugPattern,
    model: str,
    source_file: str | None,
    verified: bool,
    mode: str = "direct",
) -> dict[str, Any]:
    """Build the metadata dict for a trajectory."""
    return {
        "bug_pattern": pattern.id,
        "category": pattern.category,
        "difficulty": pattern.difficulty,
        "error_code": pattern.error_code,
        "source_file": source_file or "synthetic",
        "teacher_model": model,
        "generation_mode": mode,
        "verified": verified,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tags": pattern.tags,
    }


# ─── Verification (optional) ────────────────────────────────────────────────

def verify_fix_compiles(
    fixed_code: str,
    verify_host: str | None = None,
    timeout: int = 60,
) -> bool:
    """Optionally verify a fix compiles by writing to a temp file and running cargo check.

    If verify_host is set, runs via SSH on the remote cluster node.
    """
    # Create a minimal Cargo project structure
    test_code = f"""// Auto-generated verification
#![allow(dead_code, unused_imports, unused_variables)]
{fixed_code}
"""
    if verify_host:
        cmd = [
            "ssh", "-o", "ConnectTimeout=10", verify_host,
            f"cd /tmp && mkdir -p verify-synthetic/src && "
            f"echo '[package]\nname = \"verify\"\nversion = \"0.1.0\"\nedition = \"2021\"\n"
            f"[dependencies]\ntokio = {{ version = \"1\", features = [\"full\"] }}\n"
            f"reqwest = \"0.12\"' > verify-synthetic/Cargo.toml && "
            f"cat > verify-synthetic/src/lib.rs << 'RUSTEOF'\n{test_code}\nRUSTEOF\n"
            f"cd verify-synthetic && cargo check 2>&1",
        ]
    else:
        # Local verification
        cmd = [
            "bash", "-c",
            f"cd /tmp && mkdir -p verify-synthetic/src && "
            f"echo '[package]\nname = \"verify\"\nversion = \"0.1.0\"\nedition = \"2021\"' > verify-synthetic/Cargo.toml && "
            f"cat > verify-synthetic/src/lib.rs << 'RUSTEOF'\n{test_code}\nRUSTEOF\n"
            f"cd verify-synthetic && cargo check 2>&1",
        ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
        print(f"  Verification failed: {e}", file=sys.stderr)
        return False


def extract_rust_code_from_response(response: str) -> str | None:
    """Extract the first ```rust code block from a model response."""
    # Try ```rust first, then bare ```
    patterns = [
        r"```rust\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]
    for pat in patterns:
        match = re.search(pat, response, re.DOTALL)
        if match:
            return match.group(1).strip()
    return None


# ─── Source File Sampling ────────────────────────────────────────────────────

def find_rust_source_files(repo_root: str, max_files: int = 200) -> list[str]:
    """Find Rust source files in the repo for realistic source_file metadata."""
    result = subprocess.run(
        ["find", repo_root, "-name", "*.rs", "-not", "-path", "*/target/*",
         "-not", "-name", "mod.rs", "-not", "-name", "lib.rs", "-not", "-name", "main.rs"],
        capture_output=True, text=True, timeout=10,
    )
    files = [
        os.path.relpath(f.strip(), repo_root)
        for f in result.stdout.strip().split("\n")
        if f.strip()
    ]
    random.shuffle(files)
    return files[:max_files]


# ─── Main Pipeline ───────────────────────────────────────────────────────────

async def run_pipeline(args: argparse.Namespace) -> None:
    """Main async pipeline: generate trajectories in parallel with rate limiting."""
    # Resolve config from env and args
    api_url = args.api_url or os.environ.get("TEACHER_API_URL", "http://localhost:8317/v1")
    api_key = args.api_key or os.environ.get("TEACHER_API_KEY") or os.environ.get("SWARM_CLOUD_API_KEY", "")
    model = args.model or os.environ.get("TEACHER_MODEL", "claude-opus-4-6")
    output_path = args.output or os.environ.get("OUTPUT_PATH", "/tmp/synthetic-trajectories.jsonl")
    repo_root = args.repo_root or os.environ.get("REPO_ROOT", os.getcwd())
    num_trajectories = args.num
    mode = args.mode
    dry_run = args.dry_run
    verify = args.verify
    verify_host = args.verify_host
    max_concurrent = args.concurrency
    seed = args.seed

    # Filter patterns by category
    if args.categories and args.categories != "all":
        selected_cats = {c.strip() for c in args.categories.split(",")}
        patterns = [p for p in BUG_PATTERNS if p.category in selected_cats]
        if not patterns:
            print(f"Error: No bug patterns match categories: {selected_cats}", file=sys.stderr)
            print(f"Available categories: {', '.join(CATEGORIES)}", file=sys.stderr)
            sys.exit(1)
    else:
        patterns = list(BUG_PATTERNS)

    # Seed RNG for reproducibility
    if seed is not None:
        random.seed(seed)

    # Find real source files for metadata
    source_files = find_rust_source_files(repo_root)
    if not source_files:
        source_files = ["src/lib.rs"]
        print("Warning: No Rust source files found in repo, using 'src/lib.rs'", file=sys.stderr)

    # Build work items: cycle through patterns up to num_trajectories
    work_items: list[tuple[BugPattern, str]] = []
    for i in range(num_trajectories):
        pattern = patterns[i % len(patterns)]
        source_file = source_files[i % len(source_files)]
        work_items.append((pattern, source_file))

    # Print plan
    pattern_dist = {}
    for p, _ in work_items:
        pattern_dist[p.category] = pattern_dist.get(p.category, 0) + 1

    print(f"\n{'=' * 60}")
    print(f"Synthetic Trajectory Generation Pipeline")
    print(f"{'=' * 60}")
    print(f"  Teacher model:  {model}")
    print(f"  API endpoint:   {api_url}")
    print(f"  Mode:           {mode}")
    print(f"  Trajectories:   {num_trajectories}")
    print(f"  Bug patterns:   {len(patterns)} patterns across {len(set(p.category for p in patterns))} categories")
    print(f"  Output:         {output_path}")
    print(f"  Concurrency:    {max_concurrent}")
    print(f"  Verify:         {verify} {'(host: ' + verify_host + ')' if verify_host else ''}")
    print(f"  Dry run:        {dry_run}")
    print(f"\n  Category distribution:")
    for cat, count in sorted(pattern_dist.items()):
        print(f"    {cat:25s} {count:4d}")
    print(f"{'=' * 60}\n")

    if dry_run:
        print("[DRY RUN] Generating example trajectories without API calls...\n")

    # Rate-limiting semaphore
    semaphore = asyncio.Semaphore(max_concurrent)

    results: list[TrajectoryResult] = []
    success_count = 0
    fail_count = 0

    async with httpx.AsyncClient() as client:

        async def generate_one(idx: int, pattern: BugPattern, source_file: str) -> TrajectoryResult:
            async with semaphore:
                if mode == "agentic":
                    return await generate_agentic_trajectory(
                        client, pattern, model, api_url, api_key,
                        source_file=source_file, dry_run=dry_run,
                    )
                else:
                    return await generate_direct_trajectory(
                        client, pattern, model, api_url, api_key,
                        source_file=source_file, dry_run=dry_run,
                    )

        # Launch all tasks with progress bar
        tasks = [
            generate_one(i, pattern, source_file)
            for i, (pattern, source_file) in enumerate(work_items)
        ]

        pbar = tqdm(total=len(tasks), desc="Generating trajectories", unit="traj")
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            if result.success:
                success_count += 1
            else:
                fail_count += 1
            pbar.set_postfix(ok=success_count, fail=fail_count)
            pbar.update(1)
        pbar.close()

    # Optional: verify fixes compile
    if verify and not dry_run:
        print(f"\nVerifying {success_count} fixes compile...")
        verified_count = 0
        for result in tqdm(results, desc="Verifying", unit="fix"):
            if not result.success or not result.messages:
                continue
            # Extract the assistant's response
            assistant_msgs = [m for m in result.messages if m["role"] == "assistant"]
            if not assistant_msgs:
                continue
            last_response = assistant_msgs[-1]["content"]
            code = extract_rust_code_from_response(last_response)
            if code:
                ok = verify_fix_compiles(code, verify_host=verify_host)
                if result.metadata:
                    result.metadata["verified"] = ok
                if ok:
                    verified_count += 1
        print(f"  Verified: {verified_count}/{success_count}")

    # Write output JSONL
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    written = 0
    with open(output_path, "w") as f:
        for result in results:
            if not result.success or not result.messages:
                continue
            record = {
                "messages": result.messages,
                "metadata": result.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Pipeline Complete")
    print(f"{'=' * 60}")
    print(f"  Generated:  {success_count}/{num_trajectories}")
    print(f"  Failed:     {fail_count}/{num_trajectories}")
    print(f"  Written:    {written} records to {output_path}")

    if written > 0:
        # Print a sample
        with open(output_path) as f:
            sample = json.loads(f.readline())
        print(f"\n  Sample record metadata:")
        if sample.get("metadata"):
            for k, v in sample["metadata"].items():
                print(f"    {k}: {v}")
        print(f"  Message count: {len(sample.get('messages', []))}")

    # Compute file hash for integrity
    if written > 0:
        h = hashlib.sha256()
        with open(output_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        print(f"  SHA-256: {h.hexdigest()}")

    print(f"{'=' * 60}")

    if fail_count > 0:
        errors = [r.error for r in results if r.error]
        unique_errors = set(errors)
        print(f"\n  Unique error types ({len(unique_errors)}):")
        for err in sorted(unique_errors):
            count = errors.count(err)
            print(f"    [{count}x] {err}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training trajectories for Rust coding SFT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dry-run
  %(prog)s --num 50 --categories borrow_checker,type_mismatch
  %(prog)s --mode agentic --model gemini-3.1-pro-high --output /tmp/agentic.jsonl
  %(prog)s --verify --verify-host root@10.0.0.20 --num 20
  %(prog)s --list-patterns
  %(prog)s --list-categories

Environment variables:
  TEACHER_API_URL        API endpoint (default: http://localhost:8317/v1)
  TEACHER_API_KEY        API key (falls back to SWARM_CLOUD_API_KEY)
  TEACHER_MODEL          Model name (default: claude-opus-4-6)
  OUTPUT_PATH            Output file (default: /tmp/synthetic-trajectories.jsonl)
  REPO_ROOT              Repository root for source file sampling
  NUM_TRAJECTORIES       Number of trajectories (default: 100)
  BUG_CATEGORIES         Comma-separated categories (default: all)
        """,
    )

    parser.add_argument(
        "--num", "-n",
        type=int,
        default=int(os.environ.get("NUM_TRAJECTORIES", "100")),
        help="Number of trajectories to generate (default: 100)",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["direct", "agentic"],
        default="direct",
        help="Generation mode: 'direct' (single-turn Q&A) or 'agentic' (multi-turn tool-calling) (default: direct)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Teacher model name (default: $TEACHER_MODEL or claude-opus-4-6)",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="OpenAI-compatible API URL (default: $TEACHER_API_URL or http://localhost:8317/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (default: $TEACHER_API_KEY or $SWARM_CLOUD_API_KEY)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSONL path (default: $OUTPUT_PATH or /tmp/synthetic-trajectories.jsonl)",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root for source file sampling (default: $REPO_ROOT or cwd)",
    )
    parser.add_argument(
        "--categories", "-c",
        default=os.environ.get("BUG_CATEGORIES", "all"),
        help="Comma-separated bug categories or 'all' (default: all)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max concurrent API requests (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan and generate example outputs without calling the API",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify each fix compiles (local or via SSH)",
    )
    parser.add_argument(
        "--verify-host",
        default=None,
        help="SSH host for compilation verification (e.g., root@10.0.0.20)",
    )
    parser.add_argument(
        "--list-patterns",
        action="store_true",
        help="List all available bug patterns and exit",
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List all available categories and exit",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for teacher model (default: 0.3)",
    )

    args = parser.parse_args()

    # Handle list commands
    if args.list_categories:
        print("Available bug categories:")
        for cat in CATEGORIES:
            count = sum(1 for p in BUG_PATTERNS if p.category == cat)
            print(f"  {cat:25s} ({count} patterns)")
        sys.exit(0)

    if args.list_patterns:
        print(f"Available bug patterns ({len(BUG_PATTERNS)} total):\n")
        for p in BUG_PATTERNS:
            print(f"  {p.id:35s} [{p.category:20s}] difficulty={p.difficulty} {p.error_code or ''}")
            print(f"    {p.description}")
        sys.exit(0)

    # Validate
    if not args.dry_run and not (
        args.api_key
        or os.environ.get("TEACHER_API_KEY")
        or os.environ.get("SWARM_CLOUD_API_KEY")
    ):
        print(
            "Error: No API key provided. Set TEACHER_API_KEY, SWARM_CLOUD_API_KEY, or use --api-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()
