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

from thesdk import *

import pdb

class adc_analyser(thesdk):
    """
    Attributes
    ----------
    IOS.Members['in'].Data: ndarray, list(ndarray)
        Input signal to use for plotting. 
    plot : bool, default True
        Should the figure be drawn or not? True -> figure is drawn, False ->
        figure not drawn. 
    xlabel : string, default "Input code"
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
        self.plot = True
        self.signames = []
        self.xlabel = 'Input code'
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

        - Data is given as ascending values from LSB->MSB

        - Only one value per step
        '''
        signal = self.IOS.Members['in'].Data
        lsb_array = np.linspace(np.min(signal),np.max(signal),
                num=len(signal),endpoint=True)
        lsb_step = np.diff(lsb_array)[0]
        inl = (signal-lsb_array)/lsb_step
        inl_max = np.max(np.abs(inl))
        dnl = np.diff(signal)/lsb_step - 1
        dnl_max = np.max(np.abs(dnl))

        # Plot inl:
        code = np.arange(0,len(signal))
        text = ''
        if self.plot:
            plt.figure()
            plt.plot(code,inl)
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
    pass

