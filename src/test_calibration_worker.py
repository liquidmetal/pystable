#!/usr/bin/python

import unittest

import calibration_worker

class InvalidHeaderMockGyroData:
    def __init__(self):
        self.lineno = 1

    def __enter__(self):
        self.lineno = 1
        return self

    def readline(self):
        if self.lineno == 1:
            return 'asdfasdfutk'
        else:
            return '123123,12321232,1232123,1231232'

    def readlines(self):
        yield self.readline()

    def __exit__(self, param1, param2, param3):
        pass


class ValidMockGyroData:
    def __init__(self):
        self.lineno = 1

    def __enter__(self):
        self.lineno = 1
        return self

    def readline(self):
        if self.lineno == 1:
            self.lineno += 1
            return 'utk'
        else:
            self.lineno += 1
            return '123123,12321232,1232123,1231232'

    def readlines(self):
        if self.lineno == 10:
            return
        yield self.readline()

    def __exit__(self, param1, param2, param3):
        pass

class CalibrationWorkerTestCase(unittest.TestCase):
    def setUp(self):
        self.calibration = calibration_worker.CalibrationWorker()

    def tearDown(self):
        pass

    def test_somethingExists(self):
        self.assertTrue(self.calibration)

class GyroscopeDataFileTestCase(unittest.TestCase):
    def setUp(self):
        self.gyrodata = calibration_worker.GyroscopeDataFile('/tmp/file/doesnt/exist')
        self.badheadergyrodata = calibration_worker.GyroscopeDataFile('/tmp/file/doesnt/exist')

        def _mock_get_file_object():
            return ValidMockGyroData()

        def _mock_badheader_get_file_object():
            return InvalidHeaderMockGyroData()
        
        self.gyrodata._get_file_object = _mock_get_file_object
        self.badheadergyrodata._get_file_object = _mock_badheader_get_file_object

    def test_firstline_validation(self):
        self.gyrodata.parse()

        with self.assertRaises(calibration_worker.InvalidFileException):
            self.badheadergyrodata.parse()

class GyroscopeDataFileTestCase(unittest.TestCase):
    def setUp(self):
        self.g = calibration_worker.GyroscopeDataFile('/we/do/not/care/about/this')
        self.g.omega = {}
        self.g.omega[0] = (0, 0, 0)
        self.g.omega[100] = (1, 10, 100)

    def test_ensureEdges(self):
        self.assertEquals(self.g.fetch_closest_omega(0), (0, 0, 0))
        self.assertEquals(self.g.fetch_closest_omega(100), (1, 10, 100))

    def test_ensureInterpolation(self):
        self.assertEquals(self.g.fetch_closest_omega(25), (0.25, 2.5, 25))
        self.assertEquals(self.g.fetch_closest_omega(50), (0.50, 5.0, 50))
        self.assertEquals(self.g.fetch_closest_omega(75), (0.75, 7.5, 75))

    def test_signalFetch(self):
        self.assertEquals(self.g.get_signal_x(), [0, 1])
        self.assertEquals(self.g.get_signal_y(), [0, 10])
        self.assertEquals(self.g.get_signal_z(), [0, 100])

    def test_getTimestamps(self):
        self.assertEquals(self.g.get_timestamps(), [0, 100])

if __name__ == '__main__':
    unittest.main()
