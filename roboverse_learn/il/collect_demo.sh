## run nvidia-smi to check available GPUs
export CUDA_VISIBLE_DEVICES=0

## Force IsaacSim to exit cleanly after close (avoid shutdown hang)
export METASIM_FORCE_EXIT_ON_CLOSE=1
export METASIM_CLOSE_TIMEOUT_SEC=8

## Parameters
task_name_set=close_box
random_level=0
num_envs=1              # Number of parallel environments
demo_start_idx=0        # Index of the first demo to collect
sim_set=mujoco
cust_name=test
num_demo_success=50

expert_data_num=25

obs_space=joint_pos
act_space=joint_pos
delta_ee=0              # Delta control
extra="obs:${obs_space}_act:${act_space}"
if [ "${delta_ee}" = 1 ]; then
  extra="${extra}_delta"
fi

# # Collecting demonstration data
python ./scripts/advanced/collect_demo.py \
--sim=${sim_set} \
--task=${task_name_set} \
--num_envs=${num_envs} \
--run_unfinished \
--no-headless \
--demo_start_idx=${demo_start_idx} \
--num_demo_success ${num_demo_success} \
--cust_name=${cust_name} \
--level=${random_level}

# Convert demonstration data
python ./roboverse_learn/il/data2zarr_dp.py \
--task_name ${task_name_set}FrankaL${random_level}_${extra} \
--expert_data_num ${expert_data_num} \
--metadata_dir ./roboverse_demo/demo_${sim_set}/${task_name_set}-${cust_name}/robot-franka/success \
--action_space ${act_space} \
--observation_space ${obs_space}
