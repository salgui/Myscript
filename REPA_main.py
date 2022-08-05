## REPA MAIN ##

# Creation date: January 2022
# By: GS
# Can be used by: Anyone
# Modifiable by: Anyone (ask admins, keep track of changes)

'''
TODOS:

REPA_load_file():
    Si fenetre choisie, seulement saisir cette partie

REPA_calculus():
    determiner la fenetre de temps a partir de Configfile
            Ecrire dans result_DT la fenetre de temps
    determiner le type de cycle (min max en translation, min max en rotation)
            Ecrire dans result_DT le type de cycle

REPA_results():
    in PLOTS:
        Faire Marcher les plots sur le BOP et les variables
'''

# Modules Imports
import os
import datetime
import pandas as pd
import numpy as np
from datetime import date
from datetime import timedelta
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from PSA_modules import plot_data

def REPA_config_file():
    # Function for the USER to define the data treatment parameters
    print('')
    print('REPA_config file...')
    config = {'path': (r"\\newsol\repa\1_operacion\1_Datos_operacion\2022\2207"),
              'file': 'REPA_20220727_235958_bis_20220728_235958.csv',
        'HTF': 'HelisolXLP',
        'time_filter': False,
        'hour_f_start': 7,
        'hour_f_stop': 18,
              }

    print('Done.')
    print('')
    return config

def REPA_load_file(config):
    # Function to load select and clean the datas from REPA

    print('Loading .CSV file...')
    month = config['path'][len(config['path']) - 4:len(config['path'])]
    if int(month)<2206:
        data = pd.read_csv(config['path']+('\\')+config['file'],
        sep=';', usecols=[0, 1, 5, 18, 84, 85, 86, 87, 88, 89, 90, 103, 104, 105, 106, 108, 127, 128, 148, 155, 156, 157, 162,
        169, 170, 171, 177, 186, 187, 188, 191, 192, 193, 194, 195, 196, 197])
    else:
        data = pd.read_csv(config['path']+('\\')+config['file'],
        sep=';')
    Date = data['Date']
    Time = data['Time']

    # Formatting Date and Time
    listT = []
    for i in range(0, len(data)):
        # Creer une seule line et un clean format pour le temps
        converted_str_2time = (datetime.strptime(Time[i], '%H:%M:%S,%f'))
        Time2 = datetime.strftime(converted_str_2time, '%H:%M:%S')
        listT = np.append(listT, Time2)  # Create a list of DT with the iterative loop

    SerieT = pd.Series(listT)  # Convert from list to Serie
    # ataFrameDT=pd.DataFrame(SerieDT)   # Convert from Serie to DataFrame
    del data['Time']  # Remove the line Time with the fkg 0.582
    data.insert(1, 'Time', listT)  # Insert the line Time that we created beforehand without the fkg 0.582

    print('Done.')
    print('')

    return data,config

def REPA_Calculus(data,config):
    # Calculation of heat transfered, min, max, mean, number of cycles, etc, store everything in the dictionary result_DT

    print('Calculus...')

    end = len(data) - 1

    # Time and Date
    # Determination of the day
    Date = data['Date'][round((len(data) / 2))]
    Date = (datetime.strptime(Date, '%d/%m/%Y'))
    Date = datetime.strftime(Date, '%Y.%m.%d')

    # Determination of the time step
    time_step = datetime.strptime(data['Time'][2], '%H:%M:%S') - datetime.strptime(data['Time'][1], '%H:%M:%S')
    result_DT = {'Test day': Date, 'Time_step': str(time_step), 'Data treatment': datetime.strftime(date.today(),'%d.%m.%Y')}

    # Counting cycles
    i = 0
    cycle_counter = 0
    cycles=[0]
    ini = True
    for i in range(0, end):
        if data['RotPosition'][i] > 175:
            ini = False
        elif data['RotPosition'][i] < -15 and ini == False:
            ini = True
            cycle_counter = cycle_counter + 1
        cycles.append(cycle_counter)
    #print(cycle_counter)

    # Cycle reference (the median cycle)
    cycle_ref = round(cycle_counter / 2)
    ini = True
    i = 0
    Ref_cycle_position = []
    cycle_C = 0
    for i in range(0, end):
        if data['RotPosition'][i] > 175 and cycle_C < cycle_ref:
            ini = False
        elif data['RotPosition'][i] < -15 and ini == False and cycle_C < cycle_ref:
            ini = True
            cycle_C = cycle_C + 1
            Ref_cycle_position.append(i)  # Makes a Serie of all the i indice for when its cycling ( a cycling being a back and forth)

    data.insert(1,'executed cycles',cycles)
    result_DT.update({"Cycles": cycle_counter, 'Ref_cycle_start': Ref_cycle_position[-2], 'Ref_cycle_stop': Ref_cycle_position[-1], 'Ref_cycle':cycle_ref})

    # Determination when did the pump start and it stopped
    Pump_started = False
    if data['EY-HTF-21-W'][1] == 1:
        Pump_started = True
        Pump_start_hour = 'The pump was already on'
    else:
        i = 0
        while data['EY-HTF-21-W'][i] == 0 and i <= end - 1:
            Pump_start_hour = data['Time'][i]
            i = i + 1
            #           print(i)
        if i == end:
            Pump_start_hour = 'The pump was not started'
            Pump_stop_hour = 'The pump was not started'
            Pump_started = False
        else:
            Pump_started = True

    if Pump_started == True:
        while data['EY-HTF-21-W'][i] == 1 and i <= end - 1:
            i = i + 1
            #print(i)
        Pump_stop_hour = data['Time'][i]
        if i == end:
            Pump_stop_hour = 'The pump was still running'
    #print(Pump_start_hour)
    #print(Pump_stop_hour)

    result_DT.update({'Time_pump_start': Pump_start_hour,
                      'Time_pump_stop': Pump_stop_hour}),

    # Min Max
    # Temperature, Pressure and Flow BOP
    result_DT.update({'T_max_C': round(max(data['TT-HTF-04-W']), 2),
                      'T_min_C': round(min(data['TT-HTF-23-W']), 2),
                      'P_max_barR': round(max(data['PE-HTF-02-W']), 2),
                      'P_min_barR': round(min(data['PE-HTF-01-W']), 2),
                      'F_max_m3h': round(max(data['PE-HTF-03-W']), 2),
                      'F_min_m3h': round(min(data['PE-HTF-03-W']), 2),
                      # Temperature, Pressure and Flow EV
                      'TEV_max_C': round(max(data['TT-HTF-15-W']), 2),
                      'TEV_min_C': round(min(data['TT-HTF-15-W']), 2),
                      'PEV_max_barR': round(max(data['PE-HTF-05-W']), 2),
                      'PEV_min_barR': round(min(data['PE-HTF-05-W']), 2)}),


    # HTF properties
    # Create a module calling the capacities and flux for each line of the datas

    # UW
                        # Force
    result_DT.update({"UWFx_max_N": round(max(data['UWFx'])),
                      "UWFy_max_N": round(max(data['UWFy'])),
                      "UWFz_max_N": round(max(data['UWFz'])),
                      "UWFx_min_N": round(min(data['UWFx'])),
                      "UWFy_min_N": round(min(data['UWFy'])),
                      "UWFz_min_N": round(min(data['UWFz'])),
                        # Moment
                      "UWMx_max_Nm": round(max(data['UWMx'])),
                      "UWMy_max_Nm": round(max(data['UWMy'])),
                      "UWMz_max_Nm": round(max(data['UWMz'])),
                      "UWMx_min_Nm": round(min(data['UWMx'])),
                      "UWMy_min_Nm": round(min(data['UWMy'])),
                      "UWMz_min_Nm": round(min(data['UWMz'])),
                        # Torque
                      "UWT_min_Nm": round(min(data['TorqueSensor_West'])),
                      "UWT_max_Nm": round(max(data['TorqueSensor_West'])),
    # LW
                      # Force
                      "LWFx_max_N": round(max(data['LWFx'])),
                      "LWFy_max_N": round(max(data['LWFy'])),
                      "LWFz_max_N": round(max(data['LWFz'])),
                      "LWFx_min_N": round(min(data['LWFx'])),
                      "LWFy_min_N": round(min(data['LWFy'])),
                      "LWFz_min_N": round(min(data['LWFz'])),
                      # Moment
                      "LWMx_max_Nm": round(max(data['LWMx'])),
                      "LWMy_max_Nm": round(max(data['LWMy'])),
                      "LWMz_max_Nm": round(max(data['LWMz'])),
                      "LWMx_min_Nm": round(min(data['LWMx'])),
                      "LWMy_min_Nm": round(min(data['LWMy'])),
                      "LWMz_min_Nm": round(min(data['LWMz'])),
                      # Torque
                      "LWT_min_Nm": round(min(data['TorqueSensor_East'])),
                      "LWT_max_Nm": round(max(data['TorqueSensor_East']))}),
    print('Done.')
    print('')

    return (result_DT, data,config)

def REPA_Results(result_DT, data,config):

    print('Generating results...')

    Date = result_DT['Test day']
        ### Result and data files export

    # Create a folder if it has not been done yet
    save_folder=(config['path']+'\\'+Date+'_results')
    if os.path.exists(save_folder)==False:
       os.makedirs(save_folder)

    # export the results dictionary to .XLSX file
    result_DF = pd.DataFrame(data=result_DT, index=[0])
    result_DF = result_DF.T
    result_DF.to_excel(save_folder+'\\'+Date+'_results.xlsx')

    # export the cleaned DataFrame to .CSV file
    data.to_csv(save_folder+'\\'+Date+ '_used_data.csv')

    ### PLOTS

    ## FORCE AND MOMENTS

    # https://stackoverflow.com/questions/60046243/how-to-make-a-seaborn-distplot-for-each-column-in-a-pandas-dataframe
    # Create a vector for the Moment in UWM, UWF, LWM, LWF

    UWF = ['UWFx', 'UWFy', 'UWFz']
    UWM = ['UWMx', 'UWMy', 'UWMz']
    LWF = ['LWFx', 'LWFy', 'LWFz']
    LWM = ['LWMx', 'LWMy', 'LWMz']

    ref_cycle_range=range(result_DT['Ref_cycle_start'],result_DT['Ref_cycle_stop'])
    T_ref_cycle=round(data['TT-HTF-04-W'][round((result_DT['Ref_cycle_start']+result_DT['Ref_cycle_stop'])/2)])
    P_ref_cycle=round(data['PE-HTF-02-W'][round((result_DT['Ref_cycle_start']+result_DT['Ref_cycle_stop'])/2)])

    current='UWF'
    # Forces over a day for UWF [1/8]

    '''
    plot_data(vars1=UWF[0], vars2=UWF[1], vars3=UWF[2], vars4=None, X='Time', data=data,
              graph='YX', name=name_graph, Date='yy.mm.dd', save_folder=save_folder, dpi=800, closefig='yes')
    '''
    name_graph = (Date + '_' + str(current) + '_day.jpeg')
    plt.figure(name_graph)

    for column in UWF:
        ax = sns.lineplot(x='Time', y=column, data=data)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(len(data) / 7))
        plt.gcf().autofmt_xdate()
    plt.title('Force over time for '+current+' dyn the '+Date)
    plt.xlabel('Hour [hh:mm]')
    plt.ylabel('Force [N]')
    plt.legend(UWF)
    plt.savefig(save_folder+'\\'+name_graph)

    # Forces over a cycle for UWF [2/8]
    name_graph=(Date + '_' + current + '_cycle.jpeg')
    plt.figure(name_graph)
    for column in UWF:
        plt.plot(data['RotPosition'][ref_cycle_range],data[column][ref_cycle_range])
    plt.title('Force over a cycle for '+current+' dyn (cycle '
              +str(result_DT['Ref_cycle'])+'/'+str(result_DT['Cycles'])+') - T= '
              +str(T_ref_cycle)+'ºC P= '+str(P_ref_cycle)+' bar')
    plt.xlabel('Traverse angle (º)')
    plt.ylabel('Force [N]')
    plt.legend(UWM)
    plt.savefig(save_folder+'\\'+name_graph)

    current = 'UWM'
    # Moments over a day for UWM [3/8]
    name_graph = (Date + '_' + str(current) + '_day.jpeg')
    plt.figure(name_graph)
    for column in UWM:
        ax = sns.lineplot(x='Time', y=column, data=data)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(len(data) / 7))
        plt.gcf().autofmt_xdate()
    plt.title('Force over time for ' + current + ' dyn the ' + Date)
    plt.xlabel('Hour [hh:mm]')
    plt.ylabel('Moment [Nm]')
    plt.legend(UWF)
    plt.savefig(save_folder + '\\' + name_graph)

    # Moments over a cycle for UWM [4/8]
    name_graph=(Date + '_' + current + '_cycle.jpeg')
    plt.figure(name_graph)
    for column in UWM:
        plt.plot(data['RotPosition'][ref_cycle_range],data[column][ref_cycle_range])
    plt.title('Force over a cycle for '+current+' dyn (cycle '
              +str(result_DT['Ref_cycle'])+'/'+str(result_DT['Cycles'])+') - T= '
              +str(T_ref_cycle)+'ºC P= '+str(P_ref_cycle)+' bar')
    plt.xlabel('Traverse angle (º)')
    plt.ylabel('Moment [Nm]')
    plt.legend(UWM)
    plt.savefig(save_folder+'\\'+name_graph)


    current = 'LWF'
    # Forces over a day for LWF [5/8]
    name_graph = (Date + '_' + str(current) + '_day.jpeg')
    plt.figure(name_graph)
    for column in LWF:
        ax = sns.lineplot(x='Time', y=column, data=data)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(len(data) / 7))
        plt.gcf().autofmt_xdate()
    plt.title('Force over time for ' + current + ' dyn the ' + Date)
    plt.xlabel('Hour [hh:mm]')
    plt.ylabel('Force [N]')
    plt.legend(UWF)
    plt.savefig(save_folder + '\\' + name_graph)

    # Forces over a cycle for LWF [6/8]
    name_graph=(Date + '_' + current + '_cycle.jpeg')
    plt.figure(name_graph)
    for column in LWF:
        plt.plot(data['RotPosition'][ref_cycle_range],data[column][ref_cycle_range])
    plt.title('Force over a cycle for '+current+' dyn (cycle '
              +str(result_DT['Ref_cycle'])+'/'+str(result_DT['Cycles'])+') - T= '
              +str(T_ref_cycle)+'ºC P= '+str(P_ref_cycle)+' bar')
    plt.xlabel('Traverse angle (º)')
    plt.ylabel('Force [N]')
    plt.legend(UWM)
    plt.savefig(save_folder+'\\'+name_graph)


    current = 'LWM'
    # Moment over a day for LWM [7/8]
    name_graph = (Date + '_' + str(current) + '_day.jpeg')
    plt.figure(name_graph)
    for column in LWM:
        ax = sns.lineplot(x='Time', y=column, data=data)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(len(data) / 7))
        plt.gcf().autofmt_xdate()
    plt.title('Moment over time for ' + current + ' dyn the ' + Date)
    plt.xlabel('Hour [hh:mm]')
    plt.ylabel('Moment [Nm]')
    plt.legend(UWF)
    plt.savefig(save_folder + '\\' + name_graph)

    # Moment over a cycle for LWF [8/8]
    name_graph=(Date + '_' + current + '_cycle.jpeg')
    plt.figure(name_graph)
    for column in LWM:
        plt.plot(data['RotPosition'][ref_cycle_range],data[column][ref_cycle_range])
    plt.title('Moment over a cycle for '+current+' dyn (cycle '
              +str(result_DT['Ref_cycle'])+'/'+str(result_DT['Cycles'])+') - T= '
              +str(T_ref_cycle)+'ºC P= '+str(P_ref_cycle)+' bar')
    plt.xlabel('Traverse angle (º)')
    plt.ylabel('Moment [Nm]')
    plt.legend(UWM)
    plt.savefig(save_folder+'\\'+name_graph)

    # Torque sensor values over a day: LWT & LET

    plot_data(vars1='TorqueSensor_West',vars2='TorqueSensor_East', data=data, graph='XY',name=Date+'_torque_sensors_over_day',dpi=400,closefig='no')

    plt.title('Moment over time for ' + current + ' dyn the ' + Date)
    plt.xlabel('Hour [hh:mm]')
    plt.ylabel('Moment [Nm]')
    plt.legend('Torque sensor measurement')
    plt.savefig(save_folder + '\\' + name_graph)
    # Torque sensors over a cycle: LWT & LET
    name_graph=(Date + '_' + 'Torque_sensors' + '_cycle.jpeg')
    plt.figure(name_graph)
    for column in LWM:
        plt.plot(data['RotPosition'][ref_cycle_range],data[column][ref_cycle_range])
    plt.title('Torque over a cycle for '+current+' dyn (cycle '
              +str(result_DT['Ref_cycle'])+'/'+str(result_DT['Cycles'])+') - T= '
              +str(T_ref_cycle)+'ºC P= '+str(P_ref_cycle)+' bar')
    plt.xlabel('Traverse angle (º)')
    plt.ylabel('Torque [Nm]')
    plt.legend(UWM)
    plt.savefig(save_folder+'\\'+name_graph)


    plt.close('all')

    ## PROCESS PLOTS

    # Definition of the variables
    X=data['Time']
    Y1=data['TT-HTF-04-W']

    # Show plot
    Y2=data['PE-HTF-02-W']
    Y3=data['executed cycles']

    # Create plot

    ax1 = df.plot()
    #ax1=plt.figure()
    ax1.set_xlabel('Time [hh:mm]')
    ax1.set_ylabel('Temperature [ºC]', color='red')

    plot_1 = ax1.plot(X, Y1, color='red', label='Temp')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
     # Adding Twin Axes
    ax2 = ax1.twinx()
    ax2.set_ylabel('Pressure [bar]', color='blue')
    plot_2 = ax2.plot(X, Y2, color='blue', label='Press')
    ax2.tick_params(axis='y', labelcolor='blue')
    # Add legends
    lns = plot_1 + plot_2
    labels = [l.get_label() for l in lns]
    plt.legend(lns, labels, loc=0)




plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

index = pd.date_range(start="2020-07-01", end="2021-01-01", freq="D")
index = [pd.to_datetime(date, format='%Y-%m-%d').date() for date in index]

data = np.random.randint(1, 100, size=len(index))

df = pd.DataFrame(data=data, index=index, columns=['data'])

ax = df.plot()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))

plt.gcf().autofmt_xdate()

def REPA_main():

    # Typical main function
    print('')
    print('REPA_main execution...')
    print('')

    config = REPA_config_file()
    data,config = REPA_load_file(config)
    (result_DT, data,config) = REPA_Calculus(data,config)
    REPA_Results(result_DT, data,config)

    print('REPA_main' + ' succesfully executed')

if __name__ == "__main__":
    print()

    REPA_main()
