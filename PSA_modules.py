## PSA_modules.py

# Creation date: 16/02/2022
# By: GS
# Can be used by: Anyone
# Modifiable by: Anyone (ask admins, keep track of changes

# Descripcion: Base de modulos para funciones Python de tratamiento de datos

'''
ToDos:
Put dots to the vaporization curves to distinguish from the pressure curves
'''

def plot_data(vars1=None, vars2=None, vars3='NC', vars4='NC', X='Time', data=None,
              graph='YX', name='plot.jpeg', Date='yy.mm.dd', save_folder=None, dpi=800,closefig='yes'):
    '''
    graph='YX'
    vars3='NC'
    vars4='NC'
    vars1 = ['T_NC_in', 'T_SC_out']
    vars2 = ['P_SC_out']
    X = 'Time2'
    data = data[(result_DT['indice T reach'] - result_DT['indic start']):result_DT['indice T below'] - result_DT[
        'indic start']]
    name = Date + '_process_st'
    Date = Date
    save_folder = save_folder
    dpi=400
    '''

    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
    from matplotlib.dates import DateFormatter

    # Default parameters for plotting
    plt.rcParams['figure.figsize'] = [10, 8]

    # Diferenciate the variables type
    if vars1[0][0]=='T':
        v1_lgd='Temperature [ºC]'
    if vars1[0][0]=='P':
        v1_lgd = 'Pressure [bar]'
    if vars1[0][0]=='F':
        v1_lgd= 'Massflow [kg/s]'
    if vars1[0][0]=='L':
        v1_lgd = 'Level [mm]'
    if vars1[0][0]=='D':
        v1_lgd='DNI [W/m2]'
    if vars1[0][0]==

    if vars2[0][0]=='T':
        v2_lgd='Temperature [ºC]'
    if vars2[0][0]=='P':
        v2_lgd = 'Pressure [bar]'
    if vars2[0][0]=='F':
        v2_lgd= 'Massflow [kg/s]'
    if vars2[0][0]=='L':
        v2_lgd = 'Level [mm]'
    if vars2[0][0]=='W':
        v2_lgd='Wind speed [m/s]'

    if vars3!='NC':
        if vars3[0][0]=='T':
            v3_lgd='Temperature [ºC]'
        if vars3[0][0]=='P':
            v3_lgd = 'Pressure [bar]'
        if vars3[0][0]=='F':
            v3_lgd= 'Massflow [kg/s]'
        if vars3[0][0]=='L':
            v3_lgd = 'Level [mm]'

    if vars4 != 'NC':
        if vars4[0][0]=='T':
            v4_lgd='Temperature [ºC]'
        if vars4[0][0]=='P':
            v4_lgd = 'Pressure [bar]'
        if vars4[0][0]=='F':
            v4_lgd= 'Massflow [kg/s]'
        if vars4[0][0]=='L':
            v4_lgd = 'Level [mm]'



    if graph=='YX':
        fig, ax = plt.subplots()

    elif graph=='YYX':
        ax = plt.subplot(2, 1, 1)

    plt.title(name)

    colours=['r','g','c','k','m','y','b']

    # Data Content
    # Y1 axis
    color = 0
    for col in vars1:
        ax.plot(data[X], data[col],color=colours[color], label=col)
        color+=1

    color=len(colours)
    for col in vars2:
        ax.plot(np.nan, color=colours[color-1], label=col)
        ax.set_ylabel(v1_lgd, color='red')
        color-=1

    ax.tick_params(axis='y', labelcolor='red')
    ax.legend(loc="upper left")

    # X axis
    ax.set_xlabel('Hour [hh:mm]')
    date_form = DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_form)
    plt.gcf().autofmt_xdate()
    ax.grid(visible=True, which='major', axis='both')

    # Y2 axis
    ax2 = ax.twinx()
    ax2.set_ylabel(v2_lgd, color='k')

    color = len(colours)
    for col in vars2:
        ax2.plot(data[X], data[col], color=colours[color-1], label=col)
        ax2.tick_params(axis='y', labelcolor='k')
        color-=1

    if graph=='YYX':
        # Y3 axis
        ax3 = plt.subplot(2, 1, 2)
        color = 0
        for col in vars3:
            ax3.plot(data[X], data[col], color=colours[color], label=col)
            color += 1

        color = len(colours)
        for col in vars4:
            ax3.plot(np.nan, color=colours[color - 1], label=col)
            ax3.set_ylabel(v3_lgd, color='red')
            color -= 1

        ax3.tick_params(axis='y', labelcolor='red')
        ax3.legend(loc="upper left")

        # X2 axis
        ax3.set_xlabel('Hour [hh:mm]')
        date_form = DateFormatter("%H:%M:%S")
        ax3.xaxis.set_major_formatter(date_form)
        plt.gcf().autofmt_xdate()

        # Y4 axis
        ax4 = ax3.twinx()
        ax4.set_ylabel(v4_lgd, color='k')

        color = len(colours)
        for col in vars4:
            ax4.plot(data[X], data[col], color=colours[color - 1], label=col)
            ax4.tick_params(axis='y', labelcolor='k')
            color -= 1

    # ax2.legend(loc="upper right")
    # Plot Format
    if save_folder!=None:
        plt.savefig(save_folder + '\\' + name+'.jpeg', dpi=dpi)

    if closefig=='yes':
        plt.close()
        print(closefig)

    return ('LOL')

class HTF:
    def __init__(self,name):
        self.name=name

        if self.name == "H5A":
            self.vp = vapor_pressure(0.00000048763, 0.00010811, 0.0044446,
                                     0.15541)  # HELISOL5A vapor pressure correlation from "\\teamsites-extranet.dlr.de@SSL\DavWWWRoot\sf\stef\Freigegebene Dokumente\3_Execution\3_Technical_Information\HELISOL_5\Comparison_Properties_Helisol_vs_Syltherm.xlsx"
            self.d = density(0, 0, -0.9542,
                             941.83)  # density correlation from "\\teamsites-extranet.dlr.de@SSL\DavWWWRoot\sf\stef\Freigegebene Dokumente\3_Execution\3_Technical_Information\3_Technical_Information\2_HELISOL\HELISOL_5A"
            self.cp = cp(0, 0, 0.001724,
                         1.7077)  # Specific heat cap.[kJ / kgK] correlation with {R² = 9.9866E-01} from 50°C and up to 500 "\\teamsites-extranet.dlr.de@SSL\DavWWWRoot\sf\stef\Freigegebene Dokumente\3_Execution\3_Technical_Information\2_HELISOL\HELISOL_5A"

        if self.name == "HXLP":
            self.vp = vapor_pressure(0.000000241238, -0.0000334664, -0.000357248,
                                     0.0369922)  # HELISOLXLP vapor pressure correlation[barG](valid for T = 0 to 450°C) from \\teamsites - extranet.dlr.de @ SSL\DavWWWRoot\sf\stef\Freigegebene Dokumente\3_Execution\3_Technical_Information\2_HELISOL\HELISOL_XLP
            self.d = density(-3.7493 * pow(10, -9), 7.27837 * pow(10, -7), -0.0010141,
                             0.97076)  # density correlation in [g / cm3] for 15 bars(valid for T=0 to 450°C) from  \\teamsites-extranet.dlr.de @ SSL\DavWWWRoot\sf\stef\Freigegebene Dokumente\3_Execution\3_Technical_Information\2_HELISOL\HELISOL_XLP
            self.cp = cp(0, 0.0000016300, 0.00192339,
                         1.41889)  # Specific heat capacity correlation[kJ / kgK] for 15 bars(valid from T=35 to 450°C) from  \\teamsites-extranet.dlr.de @ SSL\DavWWWRoot\sf\stef\Freigegebene Dokumente\3_Execution\3_Technical_Information\2_HELISOL\HELISOL_XLP
        if self.name=="VP1":
            breakpoint()

class density:
    def __init__(self, d,c,b,a):
        self.d=d
        self.c=c
        self.b=b
        self.a=a
class vapor_pressure:
    def __init__(self,d,c,b,a):
        self.d=d
        self.c=c
        self.b=b
        self.a=a
class cp:
    def __init__(self,d,c,b,a):
        self.d=d
        self.c=c
        self.b=b
        self.a=a

