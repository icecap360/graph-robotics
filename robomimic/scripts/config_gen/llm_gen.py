from robomimic.scripts.config_gen.helper import *

def make_generator_helper(args):
    algo_name_short = "llm"

    generator = get_generator(
        algo_name="llm",
        config_file=os.path.join(base_path, 'robomimic/exps/templates/llm.json'),
        args=args,
        algo_name_short=algo_name_short,
        pt=True,
    )
    if args.ckpt_mode is None:
        args.ckpt_mode = "off"

    generator.add_param(
        key="train.data",
        name="ds",
        group=123456,
        values_and_names=[
            # (get_ds_cfg("ArrangeVegetables", gen_tex=False, rand_cams=False, filter_key="50_demos"), "ArrangeVegetables"),
            # (get_ds_cfg("MicrowaveThawing", gen_tex=False, rand_cams=False, filter_key="50_demos"), "MicrowaveThawing"),
            # (get_ds_cfg("RestockPantry", gen_tex=False, rand_cams=False, filter_key="50_demos"), "RestockPantry"),
            (get_ds_cfg("PreSoakPan", gen_tex=False, rand_cams=False, filter_key="50_demos"), "PreSoakPan"),
            # (get_ds_cfg("PrepareCoffee", gen_tex=False, rand_cams=False, filter_key="50_demos"), "PrepareCoffee"),
        ]
    )

    """
    ### Uncomment this code to train composite task policies ###
    generator.add_param(
        key="train.data",
        name="ds",
        group=123456,
        values_and_names=[
            (get_ds_cfg("ArrangeVegetables", gen_tex=False, rand_cams=False, filter_key="50_demos"), "ArrangeVegetables"),
            (get_ds_cfg("MicrowaveThawing", gen_tex=False, rand_cams=False, filter_key="50_demos"), "MicrowaveThawing"),
            (get_ds_cfg("RestockPantry", gen_tex=False, rand_cams=False, filter_key="50_demos"), "RestockPantry"),
            (get_ds_cfg("PreSoakPan", gen_tex=False, rand_cams=False, filter_key="50_demos"), "PreSoakPan"),
            (get_ds_cfg("PrepareCoffee", gen_tex=False, rand_cams=False, filter_key="50_demos"), "PrepareCoffee"),
        ]
    )
    generator.add_param(
        key="experiment.ckpt_path",
        name="ckpt",
        group=1389,
        values_and_names=[
            (None, "none"),
            # ("set checkpoint pth path here", "trained-ckpt"),
        ],
    )
    """
    
    """
    ### Uncomment this code to evaluate checkpoints ###
    generator.add_param(
        key="train.data,
        name="ds",
        group=1389,
        values_and_names=[
            ("set same training data as checkpoint here", "ds-name"),
        ],
    )
    generator.add_param(
        key="experiment.ckpt_path",
        name="ckpt",
        group=1389,
        values_and_names=[
            ("Add checkpoint pth path here", "trained-ckpt"),
        ],
    )
    generator.add_param(
        key="experiment.rollout.warmstart",
        name="",
        group=-1,
        values=[-1],
    )
    generator.add_param(
        key="train.num_epochs",
        name="",
        group=-1,
        values=[0],
    )
    generator.add_param(
        key="train.num_data_workers",
        name="",
        group=-1,
        values=[0],
    )
    """

    generator.add_param(
        key="train.output_dir",
        name="",
        group=-1,
        values=[
            "~/Projects/results/{env}/{mod}-{algo_name_short}".format(
                env=args.env,
                mod=args.mod,
                algo_name_short=algo_name_short,
            )
        ],
    )

    generator.add_param(
        key="train.seq_length",
        name="",
        group=-1,
        values=[10],
        hidename=True,
    )
    generator.add_param(
        key="experiment.epoch_every_n_steps",
        name="",
        group=-1,
        values=[1],
        # hidename=True,
    )
    generator.add_param(
        key="experiment.rollout.n",
        name="",
        group=-1,
        values=[1],
        # hidename=True,
    )

    return generator

if __name__ == "__main__":
    parser = get_argparser()

    args = parser.parse_args()
    make_generator(args, make_generator_helper)
