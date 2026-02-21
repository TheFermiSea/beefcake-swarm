//! Tool wrappers that add a `proxy_` prefix to tool names.
//!
//! The CLIAPIProxy (cloud proxy) prepends `proxy_` to tool names in API
//! responses. If our tools are registered as `read_file`, the model's
//! response will contain `proxy_read_file`, which rig can't match.
//!
//! Workaround: register tools with `proxy_`-prefixed names when building
//! the cloud manager. The proxy skips prefixing names that already start
//! with `proxy_`, so the round-trip is clean:
//!   tool def: `proxy_read_file` → model sees: `proxy_read_file` → calls: `proxy_read_file` ✓
//!
//! These wrappers delegate all behavior to the inner tool.

use rig::completion::ToolDefinition;
use rig::tool::Tool;

/// Generate a newtype wrapper that delegates to an inner `Tool` impl
/// but overrides `NAME` and the `definition().name` field.
macro_rules! proxy_tool {
    ($wrapper:ident wraps $inner:ty as $name:literal) => {
        pub struct $wrapper(pub $inner);

        impl Tool for $wrapper {
            const NAME: &'static str = $name;
            type Error = <$inner as Tool>::Error;
            type Args = <$inner as Tool>::Args;
            type Output = <$inner as Tool>::Output;

            async fn definition(&self, prompt: String) -> ToolDefinition {
                let mut def = self.0.definition(prompt).await;
                def.name = $name.into();
                def
            }

            async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
                self.0.call(args).await
            }
        }
    };
}

use super::fs_tools::{ListFilesTool, ReadFileTool, WriteFileTool};
use super::notebook_tool::QueryNotebookTool;
use super::patch_tool::EditFileTool;
use super::verifier_tool::RunVerifierTool;

proxy_tool!(ProxyReadFile wraps ReadFileTool as "proxy_read_file");
proxy_tool!(ProxyWriteFile wraps WriteFileTool as "proxy_write_file");
proxy_tool!(ProxyEditFile wraps EditFileTool as "proxy_edit_file");
proxy_tool!(ProxyListFiles wraps ListFilesTool as "proxy_list_files");
proxy_tool!(ProxyRunVerifier wraps RunVerifierTool as "proxy_run_verifier");
proxy_tool!(ProxyQueryNotebook wraps QueryNotebookTool as "proxy_query_notebook");
