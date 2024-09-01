"""
Config for BC algorithm.
"""

from robomimic.config.base_config import BaseConfig


class LLMConfig(BaseConfig):
    ALGO_NAME = "llm"

    def train_config(self):
        """
        BC algorithms don't need "next_obs" from hdf5 - so save on storage and compute by disabling it.
        """
        super(LLMConfig, self).train_config()
        self.train.hdf5_load_next_obs = False

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the 
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` 
        argument to the constructor. Any parameter that an algorithm needs to determine its 
        training and test-time behavior should be populated here.
        """

        # optimization parameters
        self.algo.optim_params.policy.optimizer_type = "adamw"
        self.algo.optim_params.policy.learning_rate.initial = 5e-5      # policy learning rate
        self.algo.optim_params.policy.learning_rate.decay_factor = 1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.policy.learning_rate.epoch_schedule = [] # epochs where LR decay occurs
        self.algo.optim_params.policy.learning_rate.scheduler_type = "linear" # learning rate scheduler ("multistep", "linear", etc) 
        self.algo.optim_params.policy.regularization.L2 = 0.0001          # L2 regularization strength

        self.algo.horizon = 5
        self.algo.instruction = 'Pick the pan and sponge and place them into the sink. Then turn on the sink.'
        self.algo.policies_api = {
            'PickPlaceCounterToSink':'PickPlaceCounterToSink',
            'PickPlaceSinkToCounter':'PickPlaceSinkToCounter',
            'CloseDoubleDoor':'CloseDoubleDoor',
            'TurnOnSinkFaucet':'TurnOnSinkFaucet',
            'TurnOffSinkFaucet':'TurnOffSinkFaucet',
            'PickPlaceCabinetToCounter':'PickPlaceCabinetToCounter',
            'PickPlaceCounterToCabinet':'PickPlaceCounterToCabinet',
            'PickPlaceCounterToMicrowave':'PickPlaceCounterToMicrowave',
            'PickPlaceMicrowaveToCounter':'PickPlaceMicrowaveToCounter',
            'Terminate':'Terminate',
            }
        self.algo.policies_apitype = {
            'PickPlaceCounterToSink':'PnP',
            'PickPlaceSinkToCounter':'PnP',
            'CloseDoubleDoor':'OpenClose',
            'TurnOnSinkFaucet':'OnOff',
            'TurnOffSinkFaucet':'OnOff',
            'PickPlaceCabinetToCounter':'PnP',
            'PickPlaceCounterToCabinet':'PnP',
            'PickPlaceCounterToMicrowave':'PnP',
            'PickPlaceMicrowaveToCounter':'PnP',
            'Terminate':'Terminate',
            }
        self.algo.policies_info = {
            'PickPlaceCounterToSink':'Pick an object from the counter and place it in the sink.',
            'PickPlaceSinkToCounter':'Pick an object from the sink and place it on the counter area next to the sink.',
            'CloseDoubleDoor':'Close a cabinet with two opposite-facing doors.',
            'TurnOnSinkFaucet':'Turn on the sink faucet to begin the flow of water.',
            'TurnOffSinkFaucet':'Turn off the sink faucet to begin the flow of water.',
            'PickPlaceCabinetToCounter':'Pick an object from the cabinet and place it on the counter. The cabinet is already open.',
            'PickPlaceCounterToCabinet':'Pick an object from the counter and place it inside the cabinet. The cabinet is already open.',
            'PickPlaceCounterToMicrowave':'Pick an object from the counter and place it inside the microwave. The microwave door is already open.',
            'PickPlaceMicrowaveToCounter':'Pick an object from inside the microwave and place it on the counter. The microwave door is already open.',
            'Terminate': 'End of action plan.'
            }
        self.algo.manipulator_checkpoint = '/home/qasim/Projects/results/robocasa/im-bc_xfmr/08-12-lang_cond_policy/seed_123_ds_human-50/models/model_epoch_150.pth'