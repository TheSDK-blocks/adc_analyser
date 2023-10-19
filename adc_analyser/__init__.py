"""
============
ADC_analyser
============

Calculates the INL and DNL of a given input signal
and plots the INL and DNL curves.
INL is calculated with respect to the transition points
(not the midpoints of the steps).
INL can be chosen to be calculated with endpoint or 
best-fit method.

"""
import os
import sys
if not (os.path.abspath('../../thesdk') in sys.path):
    sys.path.append(os.path.abspath('../../thesdk'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.optimize import curve_fit
from thesdk import *

import pdb

class adc_analyser(thesdk):
    """
    Attributes
    ----------
    IOS.Members['in'].Data: ndarray, list(ndarray)
        Input signal to use for plotting. 
    bits : int
        Number of bits of the ADC under test
    inl_method : string, default 'endpoint'
        Method used to calculate INL, 'endpoint' or 'best-fit'. If 'best-fit', 
        uses best-fit straight line method.
    plot : bool, default True
        Should the figure be drawn or not? True -> figure is drawn, False ->
        figure not drawn. 
    plot_method : string, default 'curve'
        Plot method, 'curve', 'bar' or 'stem'
    inl_title : string, default 'default'
        The title of the INL plot
        If 'default', title is 'INL (<inl_method>) min/max inl_min/inl_max'
    dnl_title : string, default 'default'
        The title of the DNL plot
        If 'default', title is 'DNL min/max dnl_min/dnl_max'
    sciformat : bool, default False
        Change the y-axis and annotation values to scientific format (e.g. 1e-02)
    set_ylim : bool, default False 
        Set the ylimits of the INL plot to -1.5LSB - 1.5LSB
    """
    @property
    def _classfile(self):
        return os.path.dirname(os.path.realpath(__file__)) + "/"+__name__

    def __init__(self,*arg): 
        self.print_log(type='I', msg='Initializing %s' %(__name__)) 
        self.proplist = ['inl_method', 'bits', 'plot']
        self.bits = 1
        self.inl_method = 'endpoint' 
        self.plot = True
        self.signames = []
        self.inl_title = 'default'
        self.dnl_title = 'default'
        self.sciformat = False
        self.set_ylim = False
        self.plot = True
        self.barwidth = 0.3
        self.plot_method = 'curve'

        self.IOS=Bundle()
        self.IOS.Members['in']=IO()

        self.model='py'
        self.par= False
        self.queue= []

        self.inl_endpoint = None
        self.inl_endpoint_min = None
        self.inl_endpoint_max = None
        self.inl_endpoint_abs_max = None
        self.inl_bestfit = None 
        self.inl_bestfit_min = None 
        self.inl_bestfit_max = None
        self.inl_bestfit_abs_max = None
        self.dnl = None
        self.dnl_min = None
        self.dnl_max = None
        self.dnl_abs_max = None

        if len(arg)>=1:
            parent=arg[0]
            self.copy_propval(parent,self.proplist)
            self.parent =parent;

        self.init()

    def init(self):
        pass

    def main(self):
        '''
        This module assumes:

        - Signal is given as an nsamp by 2 matrix (numpy array), 1st column is vin ramp, 2nd column output code 
          (Use e.g. numpy.column_stack( (vin_ramp, output_code) )
       
        '''
        signal = self.IOS.Members['in'].Data
        vin = signal[:,0]
        code = signal[:,1]
        bits = self.bits
        vlsb = (np.max(vin) - np.min(vin)) / 2**bits
        transition_indeces = np.where(np.diff(code))
        if len(transition_indeces[0]) < 2**bits-1:
            self.print_log(type='W',msg='Missing codes!!!')
        elif len(transition_indeces[0]) > 2**bits-1:
            self.print_log(type='W',msg='Too many transitions!!!')
        if max(np.diff(code)) > 1:
            self.print_log(type='W',msg='Codes skipped!!!')
        transition_voltages = [vin[i+1] for i in transition_indeces][0]
        offset_error = transition_voltages[0] / vlsb - 0.5    
        gain_error = ( np.max(transition_voltages) - np.min(transition_voltages) ) / vlsb - (2**bits - 2)

        signal = transition_voltages 
        
        lsb_array = np.linspace(np.min(signal),np.max(signal),
                num=len(signal),endpoint=True)
        lsb_step = np.diff(lsb_array)[0]
        inl_endpoint = (signal-lsb_array)/lsb_step
        inl_endpoint_max = np.max(inl_endpoint)
        inl_endpoint_min = np.min(inl_endpoint)
        inl_endpoint_abs_max = np.max([np.abs(inl_endpoint_min), inl_endpoint_max])
        dnl = np.diff(inl_endpoint)
        #dnl = np.diff(signal)/lsb_step - 1
        dnl_max = np.max(dnl)
        dnl_min = np.min(dnl)
        dnl_abs_max = np.max([np.abs(dnl_min), dnl_max])
        self.inl_endpoint = inl_endpoint
        self.inl_endpoint_max = inl_endpoint_max
        self.inl_endpoint_min = inl_endpoint_min
        self.inl_endpoint_abs_max = inl_endpoint_abs_max
        self.dnl = dnl
        self.dnl_max = dnl_max
        self.dnl_min = dnl_min
        self.dnl_abs_max = dnl_abs_max
        ints = np.arange(0, len(inl_endpoint))

        # Offset and gain error free transition voltages in LSB
        offset_gain_error_free = inl_endpoint + ints 
        
        if  'best-fit' in self.inl_method:

            def best_fit(x, k, b):
                return k*x + b

            popt, conv = curve_fit(best_fit, ints, offset_gain_error_free)
            k, b = popt
            bestfit_vect = [k*i + b for i in ints]
            self.inl_bestfit = offset_gain_error_free - bestfit_vect
            self.inl_bestfit_max = np.max(self.inl_bestfit)
            self.inl_bestfit_min = np.min(self.inl_bestfit)
            self.inl_bestfit_abs_max = np.max([np.abs(self.inl_bestfit_min), self.inl_bestfit_max])
            

        if self.plot:
            if 'endpoint' in self.inl_method:
                self.plot_endpoint()
            if 'best-fit' in self.inl_method:
                self.plot_bestfit()
            



    def plot_endpoint(self):
        self.plot_func(inl=self.inl_endpoint, inl_min=self.inl_endpoint_min ,inl_max=self.inl_endpoint_max, 
                dnl=self.dnl, dnl_min=self.dnl_min, dnl_max=self.dnl_max, inl_method='endpoint')

    def plot_bestfit(self):
        self.plot_func(inl=self.inl_bestfit, inl_min=self.inl_bestfit_min, inl_max=self.inl_bestfit_max, 
                dnl=self.dnl, dnl_min=self.dnl_min, dnl_max=self.dnl_max, inl_method='best-fit')


    def plot_func(self, inl, inl_min, inl_max, dnl, dnl_min, dnl_max, inl_method):

        # Plot INL:
        code = np.arange(1, len(self.inl_endpoint)+1)
        text = ''
        plt.figure()
        plt.subplots_adjust(hspace=0.9)
        plt.subplot(211)
        if self.plot_method == 'curve':
            plt.plot(code,inl)
        elif self.plot_method == 'stem':
            plt.stem(code, inl)
            plt.ylim((1.5*inl_min,1.5*inl_max))
        elif self.plot_method == 'bar':
            plt.bar(code, inl, width=self.barwidth)
        title = self.inl_title if not self.inl_title == 'default' else f'INL ({inl_method}) min/max: {inl_min:.2f}/{inl_max:.2f}'
        plt.title(title)
        plt.xlabel('Transition (k)')
        plt.ylabel('INL (LSB)')
        if self.sciformat:
            plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
        self.print_log(type='I',msg=f'Minimum INL is {inl_min}')
        self.print_log(type='I',msg=f'Maximum INL is {inl_max}')
        if self.set_ylim:
            plt.ylim((-1.5, 1.5))
        if len(code) < 10:
            plt.xticks(np.arange(np.min(code),np.max(code)+1,1.0))

        # Plot DNL:
        plt.subplot(212)
        code = np.arange(1, len(self.inl_endpoint))
        if self.plot_method == 'curve':
            plt.plot(code,dnl)
        elif self.plot_method == 'stem':
            plt.stem(code, dnl, )
            plt.ylim((1.5*dnl_min,1.5*dnl_max))
        elif self.plot_method == 'bar':
            plt.bar(code, dnl, width=self.barwidth)
        title = self.dnl_title if not self.dnl_title == 'default' else f'DNL min/max: {dnl_min:.2f}/{dnl_max:.2f}'
        plt.title(title)
        plt.xlabel('Code (k)')
        plt.ylabel('DNL (LSB)')
        if self.sciformat:
            plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
        self.print_log(type='I',msg=f'Minimum DNL is {dnl_min}')
        self.print_log(type='I',msg=f'Maximum DNL is {dnl_max}')
        if self.set_ylim:
            plt.ylim((1.5*dnl_min,1.5*dnl_max))
        if len(code) < 10:
            plt.xticks(np.arange(np.min(code),np.max(code)+1,1.0))
        plt.show(block=False)


    def run(self,*arg):
        if len(arg)>0:
            self.par=True      #flag for parallel processing
            self.queue=arg[0]  #multiprocessing.queue as the first argument
        if self.model=='py':
            self.main()
        else: 
            pass

if __name__=="__main__":
    import plot_format
    plot_format.set_style('isscc')


    # Make up some data for testing:


    # Make a random staircase curve (output code) 
    bits = 5
    nums = np.random.randint(5, 10, size=2**bits)
    code = []
    for i in range(2**bits):
        code += [i] * nums[i]

    # Make a test ramp signal (that imaginarily produced the staircase curve)
    ramp_low = 0
    ramp_high = 2
    vin_ramp = np.linspace(ramp_low, ramp_high, len(code))

    sig = np.column_stack((vin_ramp, code))    
    ana = adc_analyser() 
    ana.bits = bits
    ana.inl_method = 'endpoint', 'best-fit'
    #ana.inl_method = 'best-fit'
    #ana.inl_method = 'endpoint'
    ana.IOS.Members['in'].Data = sig
    ana.plot_method = 'curve'
    ana.run()
    ana.plot_method = 'stem'
    ana.run()
    ana.plot_method = 'bar'
    ana.run()
    input()
