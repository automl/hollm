import json
import pickle
import numpy as np
import ConfigSpace as CS


class NB201Benchmark(object):
    def __init__(
        self, path, dataset="cifar10",
        device_metric="1080ti_32_latency",
        seed=1
    ):
        cs = self.get_configuration_space()
        self.names = list(cs.values())
        self.path = path
        self.data = self._load_data()
        self.dataset = dataset
        self.device_metric = device_metric

        self.X = []
        self.y = []
        self.c = []

        self.rng = np.random.RandomState(seed)

    def _load_data(self):
        with open(self.path, "rb") as f:
            data = pickle.load(f)
        return data

    def reset_tracker(self, seed):
        # __init__() sans the data loading for multiple runs
        self.X = []
        self.y = []
        self.c = []
        self.rng = np.random.RandomState(seed)

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()
        # op_i_to_j denotes the edge that connects node i to j in the DAG representation of the NB201 search space
        cs.add(
            CS.CategoricalHyperparameter(
                "op_0_to_1",
                [
                    "none",
                    "skip_connect",
                    "avg_pool_3x3",
                    "nor_conv_1x1",
                    "nor_conv_3x3",
                ],
            )
        )
        cs.add(
            CS.CategoricalHyperparameter(
                "op_0_to_2",
                [
                    "none",
                    "skip_connect",
                    "avg_pool_3x3",
                    "nor_conv_1x1",
                    "nor_conv_3x3",
                ],
            )
        )
        cs.add(
            CS.CategoricalHyperparameter(
                "op_1_to_2",
                [
                    "none",
                    "skip_connect",
                    "avg_pool_3x3",
                    "nor_conv_1x1",
                    "nor_conv_3x3",
                ],
            )
        )
        cs.add(
            CS.CategoricalHyperparameter(
                "op_0_to_3",
                [
                    "none",
                    "skip_connect",
                    "avg_pool_3x3",
                    "nor_conv_1x1",
                    "nor_conv_3x3",
                ],
            )
        )
        cs.add(
            CS.CategoricalHyperparameter(
                "op_1_to_3",
                [
                    "none",
                    "skip_connect",
                    "avg_pool_3x3",
                    "nor_conv_1x1",
                    "nor_conv_3x3",
                ],
            )
        )
        cs.add(
            CS.CategoricalHyperparameter(
                "op_2_to_3",
                [
                    "none",
                    "skip_connect",
                    "avg_pool_3x3",
                    "nor_conv_1x1",
                    "nor_conv_3x3",
                ],
            )
        )
        return cs

    def get_best_configuration(self):
        """
        Returns the best configuration in the dataset that achieves the lowest test performance.

        :return: Returns tuple with the best configuration, its final validation performance and its test performance
        """

        configs, te, ve = [], [], []
        configs = list(sorted(self.data, key=lambda x: self.data[x][self.dataset]))
        best_config = configs[-1]
        best_test_error = 100 - self.data[best_config][self.dataset]

        return best_config, best_test_error

    def objective_function(self, config, **kwargs):
        if type(config) == CS.Configuration:
            k = json.dumps(config.get_dictionary(), sort_keys=True)
            # convert the CS representation to the nb201 string representation
            nb201_arch = "|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|".format(
                *config.get_dictionary().values()
            )
        else:
            k = json.dumps(config, sort_keys=True)
            # convert the CS representation to the nb201 string representation
            nb201_arch = "|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|".format(
                *config.values()
            )


        error = 100 - self.data[nb201_arch][self.dataset]
        time_per_minibatch = self.data[nb201_arch][self.device_metric]

        # latency in ms with a batch size of 32 on a GTX 1080Ti
        # time_per_epoch = time_per_minibatch * 50000/32 # 50000 images in CIFAR10
        # total_runtime = 200 * time_per_epoch / 1000 # in seconds; 200 epochs in total

        self.X.append(config)
        self.y.append(error)
        self.c.append(time_per_minibatch)

        return error, time_per_minibatch
