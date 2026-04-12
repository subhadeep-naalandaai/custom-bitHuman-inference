import logging

logger = logging.getLogger("heartbeat")


class HeartbeatExitReason:
    def __init__(self):
        self.is_billing_error = False
        self.is_normal_exit = True


async def initialize_heartbeat_with_retry(attributes, *args, **kwargs):
    logger.info("[HEARTBEAT] Billing heartbeat disabled (stub)")


def start_heartbeat_task(attributes, shutdown_event=None, *args, **kwargs):
    return None  # falsy → dispatcher skips cleanup_heartbeat_task


async def cleanup_heartbeat_task(heartbeat_task, *args, **kwargs):
    pass


def send_heartbeat(*args, **kwargs):
    pass
