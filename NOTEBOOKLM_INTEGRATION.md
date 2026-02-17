# The Ultimate Guide: Grounding Claude Code with NotebookLM

**Version:** 1.0
**Based on:** How To Use NotebookLM - 7 Crazy Ways (AI LABS)
**Goal:** Create an AI development workflow that is grounded, token-efficient, and hallucination-resistant.

## 📖 Introduction: The Problem & The Solution

### The Problem

AI Agents (like Claude Code) suffer from "Context Bloat":

- **Hallucinations:** Without a controlled source of truth, they guess
- **Token Waste:** Reading 50 documentation files to find one answer costs money and slows down performance
- **Amnesia:** Agents forget architectural decisions made 3 sessions ago

### The Solution

Use Google NotebookLM as an external "Second Brain" (RAG - Retrieval Augmented Generation) and Claude Code as the reasoning engine:

- NotebookLM holds the massive context (documents, codebases, research)
- Claude Code queries NotebookLM for specific answers using a CLI

## 🛠️ Phase 1: The Tech Stack Setup

You need three specific tools to build this pipeline.

### 1. Claude Code (The Agent)

Anthropic's CLI tool that allows Claude 3.7 (or latest) to run directly in your terminal, edit files, and execute commands.

**Official Link:** Anthropic Claude Code

**Installation:**

```bash
npm install -g @anthropic-ai/claude-code
```

**Auth:** Run `claude login` to authenticate with your Anthropic Console account.

### 2. NotebookLM CLI (The Bridge)

Since NotebookLM does not have a native public API (as of early 2026), developers use CLI wrappers that authenticate via browser cookies to manage notebooks programmatically.

**Recommended Tool:** NotebookLM CLI (Python) or similar open-source wrappers.

**Installation (via pip):**

```bash
pip install notebooklm-cli
```

**Authentication:**

Most CLI wrappers require a browser-based login flow:

```bash
nlm login
# or
nlm auth
```

This usually opens a Chrome window to capture the session token.

### 3. Repomix (The Packer)

A tool to "pack" your entire repository into a single, AI-readable text file/XML format. This is critical for uploading codebases to NotebookLM.

**Repository:** github.com/yamadashy/repomix

**Installation:**

```bash
npm install -g repomix
```

## ⚙️ Phase 2: Configuration (The "Brain" Link)

To make Claude use NotebookLM automatically, you must create a configuration file in your project root. This acts as the "System Prompt" for the agent.

Create a file named `claude.md` (or `.claude.md`) in your project root:

```markdown
# Agent Directives: Project Alpha

## 🧠 Memory & Knowledge Base

**PRIMARY DIRECTIVE:** Do not rely on internal training data for project specifics.
**SOURCE OF TRUTH:** Google NotebookLM (Notebook ID: `[INSERT_NOTEBOOK_ID_HERE]`)

## 🤖 Interaction Protocol

1. **Check First:** Before starting any task, query the Notebook using the `nlm` CLI to understand existing architecture.
2. **Update Loop:** When a feature is completed and tests pass, summarize the implementation and upload it to the Notebook.
3. **Research:** Do not browse the open web for generic docs. Query the specific "Research" notebooks referenced below.

## 🛠️ Tool Usage (NLM CLI)

- **Query:** `nlm query --notebook "[ID]" "Your specific question here"`
- **Add Source:** `nlm source add --notebook "[ID]" --file "path/to/doc.md"`

## 📂 Notebook Registry

- **Project Brain:** `[ID_STRING_1]` (Architecture, Decisions, State)
- **Debugging KB:** `[ID_STRING_2]` (Stack Overflow, Docs, Solutions)
- **Security:** `[ID_STRING_3]` (OWASP, CVEs, Compliance)
```

## 🚀 Phase 3: The 7 Core Strategies

### Strategy 1: The "Second Brain" (Project State)

**Use this for:** Storing architectural decisions so Claude doesn't forget.

**The Workflow:**

1. **Initialize:** Create a new Notebook in NotebookLM named "Project [Name] Brain"
2. **Link:** Paste the ID into your `claude.md`
3. **The Loop:**
   - **Planning:** "Claude, run nlm query on the Brain to fetch the requirements for the user auth module, then create a plan."
   - **Execution:** Claude writes code based on the retrieved context
   - **Documentation:** "Claude, the tests passed. Generate a summary of the auth implementation (files changed, logic used) and add it to the Brain using nlm source add."

**Why it works:** Claude's context window stays empty. It only "loads" the info it needs for the immediate task.

### Strategy 2: The Automated Research Assistant

**Use this for:** Learning new libraries without burning tokens on web scraping.

**The Workflow:**

1. **Prompt Claude:**

   > "I need to understand the 'TanStack Query v5' migration. Find the top 5 official guides and migration blog posts. Create a NEW NotebookLM notebook named 'TanStack Research', upload these URLs as sources, and then return the Notebook ID."

2. **Synthesize:**
   - Once the notebook is ready, clear Claude's context (`/clear`)
   - Prompt: "Use nlm query on the 'TanStack Research' notebook to list the 3 breaking changes that affect our useQuery hooks."

**Benefit:** You don't pay for Claude to read 50 pages of HTML. NotebookLM (Gemini 1.5 Pro) reads it for free/cheap and Claude just gets the summary.

### Strategy 3: Rapid Onboarding (Repomix)

**Use this for:** Understanding a codebase you didn't write.

**The Workflow:**

1. **Clone & Pack:**

   ```bash
   git clone https://github.com/some/legacy-repo.git
   cd legacy-repo
   repomix --style xml  # Packs the repo into repomix-output.xml
   ```

2. **Upload:**

   ```bash
   nlm source add --notebook "[ID]" --file repomix-output.xml
   ```

3. **Interrogate:**
   - Now, ask Claude: "Query the notebook to explain the relationship between the User class and the Subscription service."
   - NotebookLM searches the entire packed codebase and returns the exact logic paths

### Strategy 4: Visualizing the Invisible

**Use this for:** Creating mental maps and diagrams for the Agent.

**The Workflow:**

1. **Generate:** Ask Claude to analyze your code and generate specific data structures

   > "Analyze the /src/api folder. Generate a JSON file representing the dependency graph. Also generate a Mermaid.js flowchart of the checkout process."

2. **Store:** Save these as `dependencies.json` and `flowchart.mmd`

3. **Upload:** Push these files to your NotebookLM Brain

4. **Usage:** When Claude needs to refactor, tell it: "Check the dependencies.json source in the notebook before moving files to ensure no circular dependencies."

### Strategy 5: The Debugging Knowledge Base

**Use this for:** Fixing obscure errors without Google.

**The Workflow:**

1. **Curate:** Create a "Debugging" Notebook

2. **Populate:** Upload the following:
   - PDFs of official documentation
   - "Common Errors" pages from framework wikis
   - StackOverflow threads relevant to your specific error (print to PDF)

3. **The Fix:**
   - **Error:** `Error: Hydration failed because the initial UI does not match the render.`
   - **Prompt:** "Do NOT search Google. Query the Debugging Notebook for 'Hydration failed' and apply the recommended fix."

### Strategy 6: The Living Documentation Hub

**Use this for:** Keeping humans and agents in sync.

**The Workflow:**

1. **Write:** You (or Claude) write a `SPEC.md` for a new feature
2. **Push:** Immediately upload `SPEC.md` to a public-facing NotebookLM notebook
3. **Share:** Give the link to your non-technical Product Manager
4. **Chat:**
   - **PM:** Chats with the notebook: "Does this spec include the 'forgot password' flow?"
   - **Claude:** Queries the notebook: "Extract the validation rules for the password field from the spec."
   - **Result:** Everyone uses the exact same source of truth

### Strategy 7: The Security Handbook

**Use this for:** Automated, grounded security compliance.

**The Workflow:**

1. **Build the Handbook:** Create a "Security" Notebook

2. **Ingest Sources:**
   - OWASP Top 10 Cheat Sheet (PDF)
   - CVE Database exports for your specific dependencies (Node/Python/Go)
   - Your company's internal "Secure Coding Guidelines" PDF

3. **The Audit:**

   > "Scan payment_controller.js. Query the Security Notebook for 'SQL Injection prevention' and 'Input Sanitization'. Verify if my code adheres to the guidelines found in the notebook sources. List violations."

## 📝 Cheatsheet: Common CLI Commands

| Action | Command Pattern |
|--------|----------------|
| Login | `nlm login` |
| List Notebooks | `nlm notebook list` |
| Create Notebook | `nlm notebook create "Title"` |
| Add File Source | `nlm source add --notebook "ID" --file "doc.txt"` |
| Add URL Source | `nlm source add --notebook "ID" --url "https://..."` |
| Query (RAG) | `nlm query --notebook "ID" "Question..."` |

## ⚠️ Important Considerations

- **Privacy:** NotebookLM is a Google Cloud product. Ensure you are comfortable uploading your codebase/docs if working on proprietary software. Check your Enterprise data settings.

- **CLI Stability:** As NotebookLM evolves, CLI tools may break. Always check the GitHub repository of the CLI tool you are using for updates.

- **Token Limits:** While NotebookLM has a huge context window, passing massive amounts of text back to Claude via the terminal has limits. Ask NotebookLM for summaries and key findings, not raw data dumps.

---

*End of Guide. Generated by Gemini.*
