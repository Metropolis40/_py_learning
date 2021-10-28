


#!/usr/bin/env python3

from unittest.case import TestCase
from rearrange import rearrange_name

import unittest


class TestRearrange(unittest.TestCase) :
    def test_basic(self) :
        testcase = "wang, ada"
        expected = "ada wang"
        self.assertEqual(rearrange_name(testcase), expected)
    def test_empty(self) :
        testcase = ""
        expected = ""
        self.assertEqual(rearrange_name(testcase), expected)
    def test_double_name(self):
        testcase = "hopper, grace M."
        expected = "grace M. hopper"
        self.assertEqual(rearrange_name(testcase), expected)
    def test_one_name(self):
        testcase = "voltaire"
        expected = "voltaire"
        self.assertEqual(rearrange_name(testcase), expected)

# 这些def囊括了各种情况，例如，正常， 空值，两个名字，单个名字的情况，我们需要确保code在这些情况下都能运行，如果报错，我们就返回修改code
unittest.main()