# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os,sys
import glob
from rl_utils.atari_wrapper import make_vec_envs

def make_files_and_paths(args):
    adversary = args.adversary
    if args.save_frames:
        image_path  =  args.env_name + 'images/'

    noise_path_template = ""
    if args.robust:
        fingerprint_path = args.env_name + 'robust_fingerprint' + '/'
    else:
        fingerprint_path = args.env_name + 'fingerprint' + '/'

    noise_path_template = fingerprint_path + adversary + '_' + args.training_frames_type + str(args.training_frames) \
                + '_victim_' + args.victim_agent_mode + '_eps' + str(args.eps) + "_v{}" +'.npy'
    noise_path = fingerprint_path + adversary +'_' + args.training_frames_type + str(args.training_frames) \
                + '_victim_' + args.victim_agent_mode  + '_eps' + str(args.eps) + '.npy'
    training_data_path = args.env_name + 'train_dataset' + '/' + args.victim_agent_mode + '_train_data'
    
    if not os.path.isdir(fingerprint_path):
        os.makedirs(fingerprint_path)

    return noise_path, training_data_path, noise_path_template

def load_training_files(training_path):
    file_list_x = []
    file_list_c = []
    file_total_observations = training_path + "_total_observations.pt"

    for file in sorted(glob.glob(training_path + "_X_batch_" + "*.pt")):
        print(file)
        file_list_x.append(file)
    for file in sorted(glob.glob(training_path + "_c_batch_" + "*.pt")):
        print(file)
        file_list_c.append(file)

    print("There are {} batches in the training set".format(len(file_list_x)))

    return file_list_x, file_list_c, file_total_observations
