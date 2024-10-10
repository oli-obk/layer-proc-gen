use tracing::subscriber::set_global_default;
use tracing_subscriber::{layer::SubscriberExt as _, EnvFilter, Registry};

pub fn init_tracing() {
    let subscriber = Registry::default()
        .with(
            tracing_tree::HierarchicalLayer::new(2)
                .with_indent_lines(true)
                .with_bracketed_fields(true)
                .with_targets(true),
        )
        .with(EnvFilter::from_default_env());
    set_global_default(subscriber).unwrap();
    eprintln!();
}
