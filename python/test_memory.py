#!/usr/bin/env python

import unittest
import subprocess, os
from numpy import array
import sdm
from sdm import Bitstring

class MemoryTestCase(unittest.TestCase):
    
    def _memused(self):
        p = subprocess.Popen("ps -p %d -o rss | grep '^ *[0-9]\+ *$'" % os.getpid(), shell=True, stdout=subprocess.PIPE)
        return int(p.communicate()[0])

<<<<<<< HEAD
    #def test_saveload(self):
    #    a = Bitstring()
    #    sdm.initialize()
    #    sdm.thread_write(a, a)
    #    self.assertEqual(a.distance_to(sdm.thread_read(a)), 0)
    #    self.assertEqual(sdm.save_to_file('_test.sdm'), 0)
    #    sdm.free()
    #    sdm.initialize()
    #    self.assertTrue(a.distance_to(sdm.read(a)) > 0)
    #    sdm.free()
    #    self.assertEqual(sdm.initialize_from_file('_test.sdm'), 0)
    #    self.assertEqual(a.distance_to(sdm.thread_read(a)), 0)
    #    sdm.free()

    #def test_initialize_free(self, qty=5):
    #    m0 = self._memused()
    #    for i in range(qty):
    #        sdm.initialize()
    #        sdm.free()
    #    m1 = self._memused()
    #    self.assertTrue(m1 < 40000)
=======
    def test_saveload(self):
        a = Bitstring()
        sdm.initialize()
        sdm.thread_write(a, a)
        self.assertEqual(a.distance_to(sdm.thread_read(a)), 0)
        self.assertEqual(sdm.save_to_file('_test.sdm'), 0)
        sdm.free()
        sdm.initialize()
        self.assertTrue(a.distance_to(sdm.read(a)) > 0)
        sdm.free()
        self.assertEqual(sdm.initialize_from_file('_test.sdm'), 0)
        self.assertEqual(a.distance_to(sdm.thread_read(a)), 0)
        sdm.free()

    def test_initialize_free(self, qty=5):
        m0 = self._memused()
        for i in range(qty):
            sdm.initialize()
            sdm.free()
        m1 = self._memused()
        self.assertTrue(m1 < 40000)
<<<<<<< HEAD
>>>>>>> FETCH_HEAD
=======
>>>>>>> FETCH_HEAD

    def test_mean_distance(self, qty=10):
        sdm.initialize()
        for i in range(qty):
            a = Bitstring()
<<<<<<< HEAD
            arr = array(sdm.thread_distance(a))
=======
            arr = array(sdm.distance(a))
<<<<<<< HEAD
>>>>>>> FETCH_HEAD
=======
>>>>>>> FETCH_HEAD
            self.assertTrue(abs(arr.mean()-500) <= 1.5)
        sdm.free()

    def test_writeread(self, qty=10):
        sdm.initialize()
        for i in range(qty):
            a = Bitstring()
            sdm.thread_write(a, a)
            b = sdm.thread_read(a)
            self.assertEqual(a.distance_to(b), 0)
        sdm.free()

    def test_writereadnear(self, distance=50, qty=10):
        sdm.initialize()
        for i in range(qty):
            a = Bitstring()
            sdm.thread_write(a, a)
            b = a.copy()
            b.bitrandomswap(distance)
            c = sdm.thread_read(b)
            self.assertEqual(a.distance_to(c), 0)
        sdm.free()

<<<<<<< HEAD
=======
class MemoryThreadTestCase(unittest.TestCase):

    def test_distance(self, qty=10):
        sdm.initialize()
        for i in range(qty):
            a = Bitstring()
            d1 = sdm.distance(a)
            d2 = sdm.thread_distance(a)
            self.assertEqual(d1, d2)
        sdm.free()

    def test_radius_count(self, qty=10):
        sdm.initialize()
        for i in range(qty):
            a = Bitstring()
            cnt1 = sdm.radius_count(a)
            cnt2 = sdm.thread_radius_count(a)
            self.assertEqual(cnt1, cnt2)
        sdm.free()

>>>>>>> FETCH_HEAD
    def test_radius_count_intersect(self, qty=10):
        sdm.initialize()
        for i in range(qty):
            a = Bitstring()
            b = Bitstring()
            cnt1 = sdm.thread_radius_count_intersect(a, b)
            cnt2 = sdm.thread_radius_count_intersect(a, b)
            self.assertEqual(cnt1, cnt2)
        sdm.free()
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> FETCH_HEAD

    def test_writeread1(self, qty=20):
        sdm.initialize()
        for i in range(qty):
            a = Bitstring()
            sdm.thread_write(a, a)
            b = sdm.read(a)
            self.assertEqual(a.distance_to(b), 0)
        sdm.free()

    def test_writeread2(self, qty=10):
        sdm.initialize()
        for i in range(qty):
            a = Bitstring()
            sdm.write(a, a)
            b = sdm.thread_read(a)
            self.assertEqual(a.distance_to(b), 0)
        sdm.free()

    def test_writeread3(self, qty=10):
        sdm.initialize()
        for i in range(qty):
            a = Bitstring()
            sdm.thread_write(a, a)
            b = sdm.thread_read(a)
            self.assertEqual(a.distance_to(b), 0)
        sdm.free()
<<<<<<< HEAD
>>>>>>> FETCH_HEAD
=======
>>>>>>> FETCH_HEAD



if __name__ == '__main__':
    unittest.main()

