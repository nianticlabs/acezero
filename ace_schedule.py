import logging
from torch import optim
from torch.cuda.amp import GradScaler

_logger = logging.getLogger(__name__)

class ScheduleACE:
    """
    Handles the training schedule for the ACE model.
    """

    def __init__(self, ace_network, options):

        # Setup optimization parameters.
        self.optimizer = optim.AdamW(ace_network.parameters(), lr=options.learning_rate_min)

        if options.learning_rate_schedule not in ["circle", "constant", "1cyclepoly"]:
            raise ValueError(f"Unknown learning rate schedule: {options.learning_rate_schedule}")

        self.schedule = options.learning_rate_schedule
        self.max_iterations = options.iterations

        # Setup learning rate scheduler
        if self.schedule == 'constant':
            # No schedule. Use constant learning rate.
            self.scheduler = None

        elif self.schedule == '1cyclepoly':
            # Approximate 1cycle learning rate schedule with linear warmup and cooldown.
            self.optimizer = optim.AdamW(ace_network.parameters(), lr=options.learning_rate_max)

            # Warmup phase. Increase from warmup learning rate to max learning rate.
            self.warmup_iterations = options.learning_rate_warmup_iterations
            lr_factor_warmup = options.learning_rate_warmup_learning_rate / options.learning_rate_max
            scheduler_warmup = optim.lr_scheduler.LinearLR(self.optimizer,
                                                           start_factor=lr_factor_warmup,
                                                           total_iters=self.warmup_iterations)

            # Cooldown phase. Decrease from max learning rate to min learning rate.
            self.cooldown_trigger_percent_threshold = options.learning_rate_cooldown_trigger_percent_threshold
            self.cooldown_iterations = options.learning_rate_cooldown_iterations

            lr_factor_cooldown = options.learning_rate_min / options.learning_rate_max
            self.scheduler_cooldown = optim.lr_scheduler.LinearLR(self.optimizer,
                                                                  start_factor=1,
                                                                  end_factor=lr_factor_cooldown,
                                                                  total_iters=self.cooldown_iterations)

            # Set the scheduler to the warmup phase.
            # We will switch to cooldown scheduler when the cooldown criteria is met.
            self.scheduler = scheduler_warmup

            # Flag indicating whether we are in the final cooldown phase.
            self.in_cooldown_phase = False

            # Rolling buffer holding statistics for the cooldown criteria.
            self.cooldown_criterium_buffer = []
            # Max size of the buffer
            self.cooldown_buffer_size = 100

        else:
            # 1 Cycle learning rate schedule.
            self.optimizer = optim.AdamW(ace_network.parameters(), lr=options.learning_rate_min)
            self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                           max_lr=options.learning_rate_max,
                                                           total_steps=self.max_iterations,
                                                           cycle_momentum=False)

        # Gradient scaler in case we train with half precision.
        self.scaler = GradScaler(enabled=options.use_half)

    def check_and_set_cooldown(self, iteration):

        # cool down only supported by 1cyclepoly lr schedule
        if self.schedule != "1cyclepoly":
            return

        # check whether we are already in cool down
        if self.in_cooldown_phase:
            return

        # check whether warmup has finished, we do not want to cooldown earlier than that
        if iteration < self.warmup_iterations:
            return

        # check whether we should go into cool down according to max training duration
        start_cooldown_max_duration = iteration >= (self.max_iterations - self.cooldown_iterations)

        # check whether we should go into cool down according to dynamic criterion

        start_cooldown_dynamic = min(
            self.cooldown_criterium_buffer) > self.cooldown_trigger_percent_threshold

        if start_cooldown_max_duration or start_cooldown_dynamic:
            _logger.info(f"Starting learning rate cooldown. "
                         f"(Reason: max duration {start_cooldown_max_duration}, dynamic {start_cooldown_dynamic})")
            _logger.info(f"Training scheduled to stop in {self.cooldown_iterations} iterations.")

            self.scheduler = self.scheduler_cooldown
            self.max_iterations = iteration + self.cooldown_iterations
            self.in_cooldown_phase = True

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def backward(self, loss):
        self.scaler.scale(loss).backward()

    def step(self, batch_inliers):

        # Parameter update
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Schedule update
        if self.scheduler is not None:
            self.scheduler.step()

            if self.schedule == "1cyclepoly":

                # keep track of cooldown trigger statistic over the last n batches
                self.cooldown_criterium_buffer.append(batch_inliers)

                # trim buffer size
                if len(self.cooldown_criterium_buffer) > self.cooldown_buffer_size:
                    self.cooldown_criterium_buffer = self.cooldown_criterium_buffer[1:]
