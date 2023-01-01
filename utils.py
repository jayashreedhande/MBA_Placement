import pickle 
import json
import config2
import pandas as pd
import numpy as np

class MBAPlacement():
    def __init__(self,user_data):
        self.model_file_path = r'logistics_model.pkl'
        self.scale_model_path = r'normal_scalar.pkl'
        self.user_data = user_data
    
    def load_saved_data(self):
        with open(self.model_file_path,'rb') as f:
            self.model = pickle.load(f)

        with open(self.scale_model_path,'rb') as f:
            self.scale_model = pickle.load(f)
        
        with open('project_json_data.json','r') as f:
            self.json_data = json.load(f)

    def get_placement_prediction(self):
        self.load_saved_data()
        gender           = self.user_data['gender']
        ssc_b            = self.user_data['ssc_b']
        hsc_b            = self.user_data['hsc_b']
        hsc_s            = self.user_data['hsc_s']
        degree_t         = self.user_data['degree_t']
        workex           = self.user_data['workex']
        specialisation   = self.user_data['specialisation']

        gender          = self.json_data['gender'][gender]
        ssc_b           = self.json_data['ssc_b'][ssc_b]
        hsc_b           = self.json_data['hsc_b'][hsc_b]
        hsc_s           = self.json_data['hsc_s'][hsc_s]
        degree_t        = self.json_data['degree_t'][degree_t]
        workex          = self.json_data['workex'][workex]
        specialisation  = self.json_data['specialisation'][specialisation]


        col = len(self.json_data['columns'])
        test_array = np.zeros(col)
        test_array[0] =  gender  
        test_array[1] =  eval(self.user_data['ssc_p'])
        test_array[2] =  ssc_b
        test_array[3] =   eval(self.user_data['hsc_p']) 
        test_array[4] =  hsc_b
        test_array[5] =  hsc_s
        test_array[6] =   eval(self.user_data['degree_p'])
        test_array[7] =  degree_t
        test_array[8] =  workex
        test_array[9] =   eval(self.user_data['etest_p'])
        test_array[10] = specialisation
        test_array[11] =  eval(self.user_data['mba_p'])
       
        test_array = test_array.reshape(1,col)
        scaled_test_array = self.scale_model.transform(test_array)
        predicted_class = self.model.predict(scaled_test_array)[0]
        print('predicted_class:',predicted_class)
        return predicted_class

if __name__ =="__main__":
    place = MBAPlacement()
    place