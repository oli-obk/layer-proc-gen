use std::{cell::RefCell, future::Future, sync::Arc};

pub struct Waker;
impl std::task::Wake for Waker {
    fn wake(self: Arc<Self>) {
        todo!()
    }
}

pub struct Scheduler {
    raw_waker: Arc<Waker>,
    waker: std::task::Waker,
}

impl Default for Scheduler {
    fn default() -> Self {
        let raw_waker = Arc::new(Waker);
        Self {
            waker: raw_waker.clone().into(),
            raw_waker,
        }
    }
}

thread_local! {
    static FOO: RefCell<Option<Arc<Waker>>> = const { RefCell::new(None) };
}

impl Scheduler {
    /// Run a future synchronously and return `None` if it ends up pending.
    /// Sets up the thread local information appropriately to allow `from_tls`
    /// to work.
    pub fn try_await<T>(&self, f: impl Future<Output = T>) -> Option<T> {
        let f = std::pin::pin!(f);

        FOO.with_borrow_mut(|v| {
            assert!(v.is_none(), "nested `try_await` is not allowed");
            *v = Some(self.raw_waker.clone());
        });

        let mut futures_context = std::task::Context::from_waker(&self.waker);
        let res = match f.poll(&mut futures_context) {
            std::task::Poll::Ready(v) => Some(v),
            std::task::Poll::Pending => None,
        };
        FOO.with_borrow_mut(|v| v.take().unwrap());
        res
    }

    pub fn from_tls() -> Arc<Waker> {
        FOO.with_borrow(|v| {
            v.clone().unwrap_or_else(|| {
                panic!("cannot call `Scheduler::from_tls` outside a future awaited by `try_await`")
            })
        })
    }
}
