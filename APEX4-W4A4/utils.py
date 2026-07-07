"""
Utility functions for model training and logging.
Provides gradient scaling, norm calculation, and logging functionality.
"""
import time
import os
import sys
import time
import math
from math import inf
import logging
from termcolor import colored
import inspect
import torch


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    """
    Calculate gradient norm of an iterable of parameters.
    
    Args:
        parameters: An iterable of parameters or a single tensor
        norm_type: Type of the used p-norm (can be 'inf')
    
    Returns:
        Total norm of the parameters (viewed as a single vector)
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    
    if len(parameters) == 0:
        return torch.tensor(0.)
    
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([
                torch.norm(p.grad.detach(), norm_type).to(device) 
                for p in parameters
            ]), 
            norm_type
        )
    return total_norm


class NativeScalerWithGradNormCount:
    """
    Gradient scaler for automatic mixed precision training.
    Handles gradient scaling, unscaling, and norm calculation.
    """
    
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
        self, 
        loss, 
        optimizer, 
        clip_grad=None, 
        parameters=None, 
        create_graph=False, 
        update_grad=True, 
        retain_graph=False, 
        step=True, 
        deactive_amp=False
    ):
        """
        Scale loss, compute gradients, and optionally clip gradients.
        
        Args:
            loss: Computed loss to backpropagate
            optimizer: Optimizer for parameter updates
            clip_grad: Maximum norm for gradient clipping
            parameters: Model parameters
            create_graph: If True, graph of the derivative will be constructed
            update_grad: If True, gradients will be computed
            retain_graph: If True, graph used to compute gradients will be retained
            step: If True, optimizer step will be performed
            deactive_amp: If True, automatic mixed precision will be disabled
            
        Returns:
            Gradient norm if update_grad is True, None otherwise
        """
        if update_grad:
            if not deactive_amp:
                # Scale loss and compute gradients
                self._scaler.scale(loss).backward(
                    create_graph=create_graph, 
                    retain_graph=retain_graph
                )
                
                # Handle gradient clipping
                if clip_grad is not None:
                    assert parameters is not None
                    self._scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
                else:
                    self._scaler.unscale_(optimizer)
                    norm = ampscaler_get_grad_norm(parameters)
                
                # Perform optimization step if requested
                if step:
                    self._scaler.step(optimizer)
                self._scaler.update()
            else:
                # Standard backward pass without AMP
                loss.backward(create_graph=create_graph, retain_graph=retain_graph)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
                if step:
                    optimizer.step()
        else:
            norm = None
            
        return norm

    def state_dict(self):
        """Return the scaler's state dict."""
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        """Load the scaler's state dict."""
        self._scaler.load_state_dict(state_dict)


def create_logger(output_dir, dist_rank=0, name=''):
    """
    Create a logger with console and file handlers.
    
    Args:
        output_dir: Directory to save log files
        dist_rank: Distributed training rank
        name: Logger name
    
    Returns:
        Logger instance configured with console and file handlers
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Define log formats
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + \
                ': %(levelname)s %(message)s'

    # Add console handler for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        )
        logger.addHandler(console_handler)

    # Add file handler
    log_file = os.path.join(output_dir, f'log_rank{dist_rank}_{int(time.time())}.txt')
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S')
    )
    logger.addHandler(file_handler)

    return logger

def get_cur_info():
    """Get caller's file, line number, and function name"""
    try:
        # Get the frame of the caller (2 levels up: this function -> time() -> actual caller)
        current_frame = inspect.currentframe()
        if current_frame and current_frame.f_back and current_frame.f_back.f_back:
            caller_frame = current_frame.f_back.f_back
            return (
                os.path.basename(caller_frame.f_code.co_filename),
                caller_frame.f_lineno,
                caller_frame.f_code.co_name
            )
        else:
            return 'unknown', 0, 'unknown'
    except (ValueError, AttributeError):
        return 'unknown', 0, 'unknown'

class Timer:
    def __init__(self):  # Fixed: __ instead of **
        self.t0 = time.time()
        self.last_line = 0
        self.last_file = 'unknown'

    def time(self, debug=False):
        current_time = time.time()
        elapsed = current_time - self.t0

        if debug:
            file_name, line, frame = get_cur_info()
            print(f"TIME_INFO: FILE: {file_name} LINE: {self.last_line} to {line} spend {elapsed:.3f} s.")
            self.last_line = line
            self.last_file = file_name

        # Reset timer for next measurement
        self.t0 = current_time
        return elapsed

