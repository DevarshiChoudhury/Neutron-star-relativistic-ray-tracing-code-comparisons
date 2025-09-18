""" Instrument module for X-PSI PST + NSX-H modelling of NICER Synthetic Source  event data. """

import numpy as np
import math

import xpsi

from xpsi import Parameter
from xpsi.utils import make_verbose

class CustomInstrument(xpsi.Instrument):
    """ XTI, and XTI. """
    def construct_matrix(self):
        """ Implement response matrix parameterisation. """
        matrix = self['energy_independent_effective_area_scaling_factor'] * self.matrix
        matrix[matrix < 0.0] = 0.0

        return matrix

    def __call__(self, signal, *args):
        """ Overwrite. """

        matrix = self.construct_matrix()

        self._cached_signal = np.dot(matrix, signal)

        return self._cached_signal

    @classmethod
    @make_verbose('Loading XTI response matrix',
                  'Response matrix loaded')
    def XTI(cls,
              bounds,
              values,
              ARF,
              RMF,
              channel_energies,
              max_input,
              max_channel,
              min_input=0,
              min_channel=0,
              effective_area_scaling_factor=1.0,
              ARF_skiprows=0,
              ARF_low_column=1,
              ARF_high_column=2,
              ARF_area_column=3,
              RMF_skiprows=0,
              RMF_usecol=-1,
              channel_energies_skiprows=0,
              channel_energies_low_column=0,
              **kwargs):
        """ Load XTI instrument response matrix. """

        alpha = Parameter('energy_independent_effective_area_scaling_factor',
                          strict_bounds = (0.1,1.9),
                          bounds = bounds.get('energy_independent_effective_area_scaling_factor', None),
                          doc='XTI energy-independent effective area scaling factor',
                          symbol = r'$\alpha_{\rm XTI}$',
                          value = values.get('energy_independent_effective_area_scaling_factor',
                                             1.0 if bounds.get('energy_independent_effective_area_scaling_factor', None) is None else None))

        ARF = np.loadtxt(ARF, dtype=np.double, skiprows=ARF_skiprows)
        RMF = np.loadtxt(RMF, dtype=np.double, skiprows=RMF_skiprows, usecols=RMF_usecol)
        channel_energies = np.loadtxt(channel_energies, dtype=np.double, skiprows=channel_energies_skiprows)

        matrix = np.zeros((channel_energies.shape[0], ARF.shape[0]))

        for i in range(ARF.shape[0]):
           matrix[:,i] = RMF[i*channel_energies.shape[0]:(i+1)*channel_energies.shape[0]]

        max_input = int(max_input)
        if min_input != 0:
           min_input = int(min_input)

        edges = np.zeros(max_input - min_input + 1, dtype=np.double)

        edges[0] = ARF[min_input, ARF_low_column]; edges[1:] = ARF[min_input:max_input, ARF_high_column]

        RSP = np.zeros((max_channel - min_channel,
                       max_input - min_input), dtype=np.double)

        for i in range(RSP.shape[0]):
           RSP[i,:] = matrix[i+min_channel, min_input:max_input] * ARF[min_input:max_input, ARF_area_column] * effective_area_scaling_factor

        channels = np.arange(min_channel, max_channel)

        return cls(RSP,
                   edges,
                   channels,
                   channel_energies[min_channel:max_channel+1,channel_energies_low_column],
                   alpha, **kwargs)
