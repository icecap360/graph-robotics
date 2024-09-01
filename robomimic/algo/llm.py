"""
Implementation of Hierarchical Behavioral Cloning, where
a planner model outputs subgoals (future observations), and
an actor model is conditioned on the subgoals to try and
reach them. Largely based on the Generalization Through Imitation (GTI)
paper (see https://arxiv.org/abs/2003.06085).
"""
import textwrap
import numpy as np
from collections import OrderedDict
from copy import deepcopy
import robomimic.utils.file_utils as FileUtils
import openai
import re
import robomimic.utils.lang_utils as LangUtils
from openai import OpenAI

import torch

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config.config import Config
from robomimic.algo import register_algo_factory_func, algo_name_to_factory_func, PolicyAlgo, GL_VAE


@register_algo_factory_func("llm")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the LLM algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    return LLM, {}


class AtomicPolicy:
    def __init__(self, name, info, api, api_type, device) -> None:
        self.name = name
        self.api_type = api_type
        self.info = info
        self.api = api
        # restore policy
        # rollout_policy, self.ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)
        # self.policy = rollout_policy.policy
        # config, _ = FileUtils.config_from_checkpoint(ckpt_dict=self.ckpt_dict)
        # self.rollout_horizon = config.experiment.rollout.horizon

        # create environment from saved checkpoint
        # env, _ = FileUtils.env_from_checkpoint(
        #     ckpt_dict=ckpt_dict, 
        #     env_name=args.env, 
        #     render=args.render, 
        #     render_offscreen=(args.video_path is not None), 
        #     verbose=True,
        # )
    def get_description(self):
        if self.api_type == 'PnP':
            return self.api.strip()+'(o1) - ' + self.info.replace('an object', 'object (o1)')
        else:
            return self.api.strip() + '() - ' + self.info
    def start_episode(self):
        self.policy.start_episode()
    def get_lang(self, llm_call):
        if self.api_type == 'PnP':
            assert 'an object' in self.info
            obj_in_parentheses = re.findall(r'\((.*?)\)', llm_call)[0]
            lang = self.info.replace('an object', obj_in_parentheses)
        else:
            lang = self.info
        return lang
    def get_action(self, obs):
        return self.policy(ob=obs)

class LLM(PolicyAlgo):
    """
    Default HBC training, largely based on https://arxiv.org/abs/2003.06085
    """
    def __init__(
        self,
        algo_config,
        obs_config,
        global_config,
        obs_key_shapes,
        ac_dim,
        device,
    ):
        """
        Args:
            algo_config (Config object): instance of Config corresponding to the algo section
                of the config

            obs_config (Config object): instance of Config corresponding to the observation
                section of the config

            global_config (Config object): global training config

            obs_key_shapes (dict): dictionary that maps input/output observation keys to shapes

            ac_dim (int): action dimension

            device: torch device
        """
        self.optim_params = deepcopy(algo_config.optim_params)
        self.algo_config = algo_config
        self.obs_config = obs_config
        self.global_config = global_config
        self.ac_dim = ac_dim
        self.device = device
        self.obs_key_shapes = obs_key_shapes
        self.lang_encoder = LangUtils.LangEncoder(
            device=self.device,
        )
        self.policy_counter = 0
        openai_api_key = None 
        self.client = OpenAI(
            api_key=openai_api_key,
            # base_url="https://api.openai.com/v1"
            )
        ckpt_path = algo_config.manipulator_checkpoint
        rollout_policy, self.ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, 
                                                                          device=device, 
                                                                          verbose=True)
        self.manipulator_policy = rollout_policy.policy

        # Policies seem to have this
        # self._create_shapes(obs_config.modalities, obs_key_shapes)
        # self._create_networks()
        # self._create_optimizers()

        self.atomic_policies = {}
        for p in self.algo_config.policies_info:
            self.atomic_policies[p] = AtomicPolicy(
                p,
                self.algo_config.policies_info[p],
                self.algo_config.policies_api[p],
                self.algo_config.policies_apitype[p],
                self.device
                )
            
        task = """
            You are a robotic planner responsible for controlling a Franka Emika Panda robotic arm. You have access to a set of predefined atomic policies that the arm can execute. Your task is to suggest the next atomic policy that will make progress toward accomplishing a given instruction and, if provided, a set of atomic policies that have already been completed in the past.
            Understand the Instruction: Analyze the given instruction thoroughly to ensure you select the next atomic policy correctly.
            Ensure Completeness: If the instruction includes a partial plan, incorporate it into your choice of next suggested atomic policy.
            Use Correct Syntax: Each line in your response must accurately call one of the atomic policies, following the exact syntax provided.
            End with Terminate(): Once you think the prior plan constitutes a completion of the instruction and no further actions need to be taken, call Terminate().
            Respond with only the next atomic policy. No additional text or commentary should be included.
        """
        examples = """
        ---- Example 1 ----
        Instruction: Pick the cup and bowl from the sink and place them on the counter for drying.
        My proposed action plan is:
        PickPlaceSinkToCounter(cup)
        PickPlaceSinkToCounter(bowl)
        Terminate()

        ---- Example 2 ----
        Instruction: Open the cabinet. Pick the cleaner and place it on the counter. Then close the cabinet.		
        My proposed action plan is:
        OpenDoubleDoor()
        PickPlaceCabinetToCounter(cleaner)
        CloseDoubleDoor()
        Terminate()

        ---- Example 3 ----
        Instruction: Pick the sponge from the counter and place it in the sink.		
        My proposed action plan is:
        PickPlaceCounterToSink(sponge)
        Terminate()
        """
        system_prompt = '---- Task ----' + task + '\n'
        atomic_policy_prompt = '\n'.join([p.get_description() for p in self.atomic_policies.values()])
        system_prompt += '---- Available Atomic Policies ----' + atomic_policy_prompt + '\n'
        system_prompt += '---- Examples ----' + examples + '\n'
        self.system_prompt = system_prompt

        self.instruction =  algo_config.instruction + '\nMy proposed action plan is:\n'
        self.active_policy = None
        self.active_policy_done = False
        self.horizon = 5
        self.LLM_cache = {}
        ### BC Transformer params
        self.context_length = 10
        self.supervise_all_steps = True
        self.pred_future_acs = True
        if self.pred_future_acs:
            assert self.supervise_all_steps is True

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        info = OrderedDict()
        return info
        if self.active_policy_done:
            action = gpt4_suggest_action(self.system_prompt, self.instruction, self.client)
            api_call = action.split('(')[0]
            name = None
            for (k,v) in self.atomic_policies.items():
                if v.api == api_call:
                    name = v.name
                    break
            self.active_policy_done = False
            self.active_policy = self.atomic_policies[name]
            self.instruction += action + '\n'
            lang_policy = self.active_policy.get_lang(action)
            self.lang_emb = self.lang_encoder.get_lang_emb(lang_policy)
        # Call manipulator policy (there is only 1), with the updated lang_emb
        batch['obs']['lang_emb'] = self.lang_emb.repeat((1,10,1))
        info = self.manipulator_policy.train_on_batch(batch, epoch, validate)
        # info["predictions"] = TensorUtils.detach(manipulator_ret['predictions'])
        # info['losses'] = TensorUtils.detach(torch.Tensor(0.0, device=self.device))
        return OrderedDict() # info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """

        info["Loss"] = 0.0 # TensorUtils.detach(torch.Tensor(0.0, device=self.device))
        #  TensorUtils.detach(info['losses'])
        return info

    def on_epoch_end(self, epoch):
        """
        Called at the end of each epoch.
        """
        self.manipulator_policy.on_epoch_end(epoch)

    def set_eval(self):
        """
        Prepare networks for evaluation.
        """
        self.manipulator_policy.set_eval()
        
    def set_train(self):
        """
        Prepare networks for training.
        """
        # return
        self.manipulator_policy.set_train()

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        x =  {'manipulator_policy':self.manipulator_policy.serialize()}
        return x

    def deserialize(self, model_dict):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
        """
        self.manipulator_policy = model_dict['manipulator_policy']

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        # Removed TensorUtils.detach
        
        if self.active_policy_done:
            action = gpt4_suggest_action(self.system_prompt, self.instruction, self.client)
            print("\n ---- New LLM Suggested policy iter{x} ----".format(x=self.policy_counter))
            print(action)
            api_call = action.split('(')[0]
            name = None
            for (k,v) in self.atomic_policies.items():
                if v.api == api_call:
                    name = v.name
                    break
            self.active_policy_done = False
            self.manipulator_policy.reset()
            self.active_policy = self.atomic_policies[name]
            self.instruction += action + '\n'
            lang_policy = self.active_policy.get_lang(action)
            print( lang_policy)
            self.lang_emb = self.lang_encoder.get_lang_emb(lang_policy)
        # Call manipulator policy (there is only 1), with the updated lang_emb
        obs_dict['lang_emb'] = self.lang_emb.repeat((1,10,1))
        self.policy_counter += 1
        if self.policy_counter % 1500 == 0:
            self.active_policy_done = True
        if 'terminate' in self.active_policy.name.lower():   
            return torch.zeros((1,12), device=self.device)
        else:
            return self.manipulator_policy.get_action(obs_dict)

        info["predictions"] = TensorUtils.detach(manipulator_ret['predictions'])
        info['losses'] = TensorUtils.detach(torch.Tensor(0.0, device=self.device))

        raise NotImplementedError()
        return self.atomic_policies[0].policy.get_action(obs_dict = obs_dict)

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        self.active_policy_done = True
        self.manipulator_policy.reset()

    def __repr__(self):
        """
        Pretty print algorithm and network description.
        """
        msg = str(self.__class__.__name__)
        return msg
        # return msg + "Planner:\n" + textwrap.indent(self.planner.__repr__(), '  ') + \
            #    "\n\nPolicy:\n" + textwrap.indent(self.actor.__repr__(), '  ')
    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        h = self.context_length
        input_batch["obs"] = {k: batch["obs"][k][:, :h, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present

        if self.supervise_all_steps:
            # supervision on entire sequence (instead of just current timestep)
            if self.pred_future_acs:
                ac_start = h - 1
            else:
                ac_start = 0
            input_batch["actions"] = batch["actions"][:, ac_start:ac_start+h, :]
        else:
            # just use current timestep
            input_batch["actions"] = batch["actions"][:, h-1, :]

        if self.pred_future_acs:
            assert input_batch["actions"].shape[1] == h

        input_batch = TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
        return input_batch



#@title LLM Scoring
def gpt4_call(system, user_assistant, client):
    assert isinstance(system, str), "`system` should be a string"
    assert isinstance(user_assistant, list), "`user_assistant` should be a list"
    # system_msg = [{"role": "system", "content": system}]
    # user_assistant_msgs = [
    #   {"role": "assistant", "content": user_assistant[i]} if i % 2 else {"role": "user", "content": user_assistant[i]}
    #   for i in range(len(user_assistant))]  
    # msgs = system_msg + user_assistant_msgs
    # response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
    #                                       messages=msgs)
    # status_code = response["choices"][0]["finish_reason"]
    # assert status_code == "stop", f"The status code was {status_code}."
    # return response["choices"][0]["message"]["content"]
    chat_completion = client.chat.completions.create(
        messages=[{"role": 'system', "content": system},
                  {"role":'user', "content": user_assistant[0]}
        ],
        model="gpt-4o-mini",
        logprobs=True
        )
    return chat_completion.choices[0]

def gpt4_suggest_action(system_prompt, instruction, client):
    response = gpt4_call(system_prompt, [instruction], client)
    return response.message.content.split('\n')[0]

def gpt4_scoring(system_prompt, instruction, limit_num_options=None, print_tokens=False):
    if limit_num_options:
        options = options[:limit_num_options]
    response = gpt4_call(system_prompt, instruction)
    conf = np.mean([token.logprob for token in response.logprobs.content])
    raise NotImplementedError
    # scores = {}
    # for option, choice in zip(options, response["choices"]):
    #     tokens = choice["logprobs"]["tokens"]
    #     token_logprobs = choice["logprobs"]["token_logprobs"]

    #     total_logprob = 0
    #     for token, token_logprob in zip(reversed(tokens), reversed(token_logprobs)):
    #     print_tokens and print(token, token_logprob)
    #     if option_start is None and not token in option:
    #         break
    #     if token == option_start:
    #         break
    #     total_logprob += token_logprob
    #     scores[option] = total_logprob

    # for i, option in enumerate(sorted(scores.items(), key=lambda x : -x[1])):
    #     verbose and print(option[1], "\t", option[0])
    #     if i >= 10:
    #     break

    # return scores, response
