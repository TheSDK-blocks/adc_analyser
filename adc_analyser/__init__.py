"""
============
DAC_analyser
============

Calculates the INL and DNL of a given input signal
and plots the INL curve

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
    inl_method : string, default 'endpoint'
        Method used to calculate INL, 'endpoint' or 'best-fit'. If 'best-fit', 
        uses best-fit straight line method
    plot : bool, default True
        Should the figure be drawn or not? True -> figure is drawn, False ->
        figure not drawn. 
    title : string, default 'default'
        The title of the figure
    xlabel : string, default 'Transition (k)'
        The xlabel of the figure
    ylabel : string, default "INL (LSB)"
        The ylabel of the figure
    annotate : bool, default True
        Add maximum INL and maximum DNL to the INL curve figure
    sciformat : bool, default True
        Change the y-axis and annotation values to scientific format (e.g. 1e-02)
    set_ylim : bool, default True
        Set the ylimits of the curve to -1.5LSB - 1.5LSB
    """
    @property
    def _classfile(self):
        return os.path.dirname(os.path.realpath(__file__)) + "/"+__name__

    def __init__(self,*arg): 
        self.print_log(type='I', msg='Initializing %s' %(__name__)) 
        self.proplist = [ ]
        self.Nbits = 1
        self.inl_method = 'endpoint' 
        self.plot = True
        self.signames = []
        self.title = 'default'
        self.xlabel = 'Transition (k)'
        self.ylabel = 'INL (LSB)'
        self.annotate = True
        self.sciformat = True
        self.set_ylim = True
        self.plot = True

        self.IOS=Bundle()
        self.IOS.Members['in']=IO()

        self.model='py'
        self.par= False
        self.queue= []

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
       
        '''
        signal = self.IOS.Members['in'].Data
        v = signal[:,0]
        code = signal[:,1]
        Nbits = self.Nbits
        vlsb = (np.max(v) - np.min(v)) / 2**Nbits
        transition_indeces = np.where(np.diff(code))
        if len(transition_indeces) < 2**Nbits-1:
            self.print_log(type='I',msg='Missing codes!!!')
        elif len(transition_indeces) > 2**Nbits-1:
            self.print_log(type='I',msg='Too many transitions!!!')
        if max(np.diff(code)) > 1:
            self.print_log(type='I',msg='Codes skipped!!!')
        transition_voltages = [v[i+1] for i in transition_indeces][0]
        offset_error = transition_voltages[0] / vlsb - 0.5    
        gain_error = ( np.max(transition_voltages) - np.min(transition_voltages) ) / vlsb - (2**Nbits - 2)

        signal = transition_voltages 
        
        lsb_array = np.linspace(np.min(signal),np.max(signal),
                num=len(signal),endpoint=True)
        lsb_step = np.diff(lsb_array)[0]
        inl_endpoint = (signal-lsb_array)/lsb_step
        inl_endpoint_max = np.max(np.abs(inl_endpoint))
        dnl = np.diff(inl_endpoint)
        #dnl = np.diff(signal)/lsb_step - 1
        dnl_max = np.max(np.abs(dnl))
        pdb.set_trace()
        ints = np.arange(0, len(inl_endpoint))

        # Offset and gain error free transition voltages in LSB
        offset_gain_error_free = inl_endpoint + ints 
        
        if self.inl_method == 'best-fit':

            def best_fit(x, k, b):
                return k*x + b

            popt, conv = curve_fit(best_fit, ints, offset_gain_error_free)
            k, b = popt
            bestfit_vect = [k*i + b for i in ints]
            inl_bestfit = offset_gain_error_free - bestfit_vect
            inl_bestfit_max = np.max(np.abs(inl_bestfit))
            inl = inl_bestfit
            inl_max = inl_bestfit_max

        else:
            inl = inl_endpoint
            inl_max = inl_endpoint_max
         
        

        # Plot inl:
        code = np.arange(1, len(signal)+1)
        text = ''
        if self.plot:
            plt.figure()
            plt.plot(code,inl)
            title = self.title if not  self.title == 'default' else f'INL ({self.inl_method})'
            plt.title(title)
            plt.xlabel(self.xlabel)
            plt.ylabel(self.ylabel)
            if self.sciformat:
                text+='Max INL = {:.2e}\n'.format(inl_max)
                text+='Max DNL = {:.2e}'.format(dnl_max)
                self.print_log(type='I',msg=f'Maximun INL is {inl_max}')
                self.print_log(type='I',msg=f'Maximun DNL is {dnl_max}')
                plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
            else:
                text+='Max INL = {:.4f}\n'.format(inl_max)
                text+='Max DNL = {:.4f}'.format(dnl_max)
                self.print_log(type='I',msg=f'Maximun INL is {inl_max}')
                self.print_log(type='I',msg=f'Maximun DNL is {dnl_max}')
            if self.annotate:
                plt.text(0.025,0.975,text,usetex=plt.rcParams['text.usetex'],
                        horizontalalignment='left',verticalalignment='top',
                        multialignment='left',fontsize=plt.rcParams['legend.fontsize'],
                        fontweight='normal',transform=plt.gca().transAxes,
                        bbox=dict(boxstyle='square,pad=0',fc='#ffffffa0',ec='none'))
            if self.set_ylim:
                plt.ylim((-1.5*inl_max,1.5*inl_max))
            if len(code) < 10:
                plt.xticks(np.arange(np.min(code),np.max(code)+1,1.0))
            plt.show(block=False)
            return inl, dnl
        else:
            return inl, dnl


    def run(self,*arg):
        if len(arg)>0:
            self.par=True      #flag for parallel processing
            self.queue=arg[0]  #multiprocessing.queue as the first argument
        if self.model=='py':
            self.main()
        else: 
            pass

if __name__=="__main__":
    code = [0]*50 + [1] * 45 +[2]*55 + [3] * 50 +[4]*50 + [5] * 50 +[6]*50 + [7] * 50 + [7]*400
    ramp_low = 0
    ramp_high = 2
    v = np.linspace(ramp_low, ramp_high, len(code))

    sig = np.column_stack((v,code))    
    ana = adc_analyser() 
    ana.Nbits = 3
    ana.inl_method = 'endpoint'
    #ana.inl_method = 'best-fit'
    ana.IOS.Members['in'].Data = sig
    ana.run()
    input()
