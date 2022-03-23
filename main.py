import pandas as pd
from joblib import load

def get_prediction(loss_of_capacity, managnese_cathode_voltage_conduction, temperature_during_high_performance, cobalt_anode_voltage_conduction, internal_resistance, internal_humidity):
    val = [[loss_of_capacity,managnese_cathode_voltage_conduction,temperature_during_high_performance,cobalt_anode_voltage_conduction,internal_resistance,internal_humidity]]
    data = pd.DataFrame(val, columns=['loss_of_capacity', 'managnese_cathode_voltage_conduction', 'temperature_during_high_performance', 'cobalt_anode_voltage_conduction', 'internal_resistance', 'internal_humidity'])
    data['lc'] = 0
    data.lc[(data['loss_of_capacity'] > 79) & (data['loss_of_capacity'] <= 84)] = 1
    data.lc[(data['loss_of_capacity'] > 84) & (data['loss_of_capacity'] <= 90)] = 2
    data.lc[(data['loss_of_capacity'] > 90)] = 3

    data['mcvc'] = 0
    data.mcvc[(data['managnese_cathode_voltage_conduction'] > 40) & (data['managnese_cathode_voltage_conduction'] <= 43)] = 1
    data.mcvc[(data['managnese_cathode_voltage_conduction'] > 43) & (data['managnese_cathode_voltage_conduction'] <= 46)] = 2
    data.mcvc[(data['managnese_cathode_voltage_conduction'] > 46)] = 3

    data['tdhp'] = 0
    data.tdhp[(data['temperature_during_high_performance'] < 60) & (data['temperature_during_high_performance'] >= 50)] = 1
    data.tdhp[(data['temperature_during_high_performance'] < 50) & (data['temperature_during_high_performance'] >= 45)] = 2
    data.tdhp[(data['temperature_during_high_performance'] < 45)] = 3

    data['cavc'] = 0
    data.cavc[(data['cobalt_anode_voltage_conduction'] > 40) & (data['cobalt_anode_voltage_conduction'] <= 43)] = 1
    data.cavc[(data['cobalt_anode_voltage_conduction'] > 43) & (data['cobalt_anode_voltage_conduction'] <= 46)] = 2
    data.cavc[(data['cobalt_anode_voltage_conduction'] > 46)] = 3

    data['ir'] = 0
    data.lc[(data['internal_resistance'] > 80) & (data['internal_resistance'] <= 110)] = 1
    data.lc[(data['internal_resistance'] > 110) & (data['internal_resistance'] <= 160)] = 2
    data.lc[(data['internal_resistance'] > 160)] = 3

    data['ih'] = 0
    data.lc[(data['internal_humidity'] < 70) & (data['internal_humidity'] >= 55)] = 1
    data.lc[(data['internal_humidity'] < 55) & (data['internal_humidity'] >= 30)] = 2
    data.lc[(data['internal_humidity'] < 30)] = 3

    data = data.drop(['loss_of_capacity', 'managnese_cathode_voltage_conduction', 'temperature_during_high_performance', 'cobalt_anode_voltage_conduction', 'internal_resistance', 'internal_humidity'], axis=1)

    model = load('model.joblib')
    scale = load('normalize.joblib')

    X = data.values

    scale.transform(X)

    return pd.DataFrame([model.predict(X)], columns=['Target'])