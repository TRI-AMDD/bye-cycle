import unittest
import bye_cycle
import numpy as np
import sys

class TestFunctionality(unittest.TestCase):
    """Tests code functionality for `bye_cycle`."""
    def test_imports(self):
        try:
            import bye_cycle, tensorflow, beep, pandas, sklearn
        except ImportError as error:
            raise error

class TestUtils(unittest.TestCase):
    """Tests for `bye_cycle.utils` methods."""
    def test_clean_cycle_data(self):
        from beep.structure.cli import auto_load_processed
        test_cell = auto_load_processed("./test_files/test_cell_LFP.json")
        columns=['voltage', 'current', 'cycle_index', 'discharge_capacity']
        cycle_index = 5
        clean_structured_data_tested_cell = bye_cycle.clean_cycle_data(test_cell,  cycle_index,
                                                             columns=['voltage', 'current', 'cycle_index', 'discharge_capacity'])
        assert list(clean_structured_data_tested_cell.columns) == columns
        assert np.unique(clean_structured_data_tested_cell['cycle_index']) == cycle_index

if __name__ == '__main__':
    unittest.main()