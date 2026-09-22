import json
import unittest

from trainingfree.queue_bf16 import launch_environment, select_devices, select_model_and_tool_devices


class ToolGpuSelectionTests(unittest.TestCase):
    def test_default_keeps_original_weight_only_selection_and_budget(self):
        rows = [(0,10000),(1,10000),(2,3000)]
        selected,tool = select_model_and_tool_devices(rows)
        self.assertEqual(selected,select_devices(rows))
        self.assertIsNone(tool)
        base = {'OTHER_SETTING':'keep','CVPR_GPU_BUDGET_MIB':'old'}
        env = launch_environment(selected,tool,base)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'0,1')
        self.assertEqual(json.loads(env['CVPR_GPU_BUDGETS_MIB']),{'0':9744,'1':9744})
        self.assertNotIn('CVPR_GPU_BUDGET_MIB',env)
        self.assertNotIn('CVPR_TOOL_DEVICE',env)
        self.assertNotIn('CVPR_TOOL_GPU_BUDGET_MIB',env)
        self.assertEqual(base,{'OTHER_SETTING':'keep','CVPR_GPU_BUDGET_MIB':'old'})

    def test_lowest_sufficient_spare_card_goes_to_tools(self):
        rows = [(0,10000),(1,10000),(2,4000),(3,3000)]
        selected,tool = select_model_and_tool_devices(rows,tool_gpu_free_mib=2800)
        self.assertEqual(tool,(3,3000))
        self.assertEqual(selected,[(0,10000),(1,10000)])
        self.assertNotIn(tool,selected)
        # Tools may fit a card too small for select_devices' shard minimum.
        self.assertEqual(select_devices([(3,3000)],weight_mib=1),[])

    def test_model_and_tools_must_both_fit_without_sharing_card(self):
        self.assertEqual(select_model_and_tool_devices([(0,22000)],tool_gpu_free_mib=2000),([],None))
        self.assertEqual(select_model_and_tool_devices([(0,10000),(1,10000),(2,4000)],
                                                       tool_gpu_free_mib=5000),([],None))
        self.assertEqual(select_model_and_tool_devices([(0,6000),(1,4000),(2,3000)],
                                                       tool_gpu_free_mib=2000),([],None))

    def test_model_budget_cannot_include_tool_gpu(self):
        selected,tool = select_model_and_tool_devices([(4,10000),(7,10000),(9,4000)],
                                                       tool_gpu_free_mib=3500)
        env = launch_environment(selected,tool,{'CVPR_GPU_BUDGET_MIB':'old'})
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'4,7,9')
        self.assertEqual(env['CVPR_TOOL_DEVICE'],'cuda:2')
        self.assertEqual(env['CVPR_TOOL_GPU_BUDGET_MIB'],'3744')
        budgets = json.loads(env['CVPR_GPU_BUDGETS_MIB'])
        self.assertEqual(budgets,{'0':9744,'1':9744})
        self.assertNotIn('2',budgets)

    def test_smallest_sufficient_threshold_ties_and_invalid_request(self):
        rows = [(0,22000),(3,4000),(2,4000),(1,2000)]
        selected,tool = select_model_and_tool_devices(rows,tool_gpu_free_mib=4000)
        self.assertEqual(selected,[(0,22000)])
        self.assertEqual(tool,(2,4000))
        with self.assertRaises(ValueError):
            select_model_and_tool_devices(rows,tool_gpu_free_mib=-1)
        self.assertEqual(select_model_and_tool_devices([(0,22000),(1,200)],
                                                       tool_gpu_free_mib=1),([],None))

    def test_disabled_tool_mode_clears_stale_tool_mapping(self):
        env = launch_environment([(0,22000)],None,
                                 {'CVPR_TOOL_DEVICE':'cuda:3','CVPR_TOOL_GPU_BUDGET_MIB':'4096'})
        self.assertNotIn('CVPR_TOOL_DEVICE',env)
        self.assertNotIn('CVPR_TOOL_GPU_BUDGET_MIB',env)


if __name__ == '__main__':
    unittest.main()
