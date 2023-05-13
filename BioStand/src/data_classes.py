# class Sets():
#     #options
#     control_mode = 'Displacemnet'
#     control_function = 'Ramp'
#     magnitude = 0
#     magnitude_percent_X = 0
#     magnitude_percent_Y = 0
#     preload_magnitude_X = 0
#     preload_magnitude_Y = 0
#     stretch_phase_time = 0
#     hold_phase_time = 0
#     recover_phase_time = 0
#     rest_phase_time = 0
#     cycle_count = 1
#     delta_time = 1
#     result={}
#     #can be replace by previous mesurement
#     result_time = 0
    
#     #func prepearing data to send to controller
#     def prepare_data(self):
#         #stretch lenght 
#         mag_x = self.magnitude/100*self.magnitude_percent_X
#         mag_y = self.magnitude/100*self.magnitude_percent_Y
        
#         #cycle_time = self.stretch_phase_time+self.hold_phase_time+self.recover_phase_time+self.rest_phase_time
        
#         stretch_speed_x = mag_x/self.stretch_phase_time
#         stretch_speed_y = mag_y/self.stretch_phase_time
#         recover_speed_x = mag_x/self.recover_phase_time
#         recover_speed_y = mag_y/self.recover_phase_time
#         #saves end time
#         full_time = self.result_time
        
        
#         for counta in range (0,self.cycle_count):
#             #stretch phase            
#             for t in range(1,self.stretch_phase_time+self.delta_time,self.delta_time):
#                 self.result[full_time + t] = (((stretch_speed_x*self.delta_time)/2,(stretch_speed_x*self.delta_time)/2),
#                                   ((stretch_speed_y*self.delta_time)/2,(stretch_speed_y*self.delta_time)/2))
#             full_time = full_time + self.stretch_phase_time  
#             #hold phase
#             for t in range(1,self.hold_phase_time+self.delta_time,self.delta_time):
#                 self.result[full_time + t] = ((0,0),(0,0))
#             full_time = full_time + self.hold_phase_time
#             #recover phase    
#             for t in range(1,self.recover_phase_time+self.delta_time,self.delta_time):
#                 self.result[full_time + t] = (((-recover_speed_x*self.delta_time)/2,(-recover_speed_x*self.delta_time)/2),
#                                   ((-recover_speed_y*self.delta_time)/2,(-recover_speed_y*self.delta_time)/2))
#             full_time = full_time + self.recover_phase_time
#             #rest phase                
#             for t in range(1,self.rest_phase_time+self.delta_time,self.delta_time):
#                 self.result[full_time + t] = ((0,0),(0,0))   
                              
#             full_time = full_time + self.rest_phase_time    
#             self.result_time = full_time



class Sets():
    #options
    control_mode = 'Displacemnet'
    control_function = 'Ramp'
    magnitude = 0
    magnitude_percent_X = 0
    magnitude_percent_Y = 0
    preload_magnitude_X = 0
    preload_magnitude_Y = 0
    stretch_phase_time = 0
    hold_phase_time = 0
    recover_phase_time = 0
    rest_phase_time = 0
    cycle_count = 1
    delta_time = 1
    result={}
    #can be replace by previous mesurement
    result_time = 0
    #tails summarize to predict errors
    
    stretch_err_x1 = 0
    stretch_err_y1 = 0
    recovery_err_x1 = 0
    recovery_err_y1 = 0 
    
    stretch_err_x2 = 0
    stretch_err_y2 = 0
    recovery_err_x2 = 0
    recovery_err_y2 = 0 
    
    #func prepearing data to send to controller
    def prepare_data(self):
        
        def fix (num,err):
            num, tail = str(num).split('.')
            #print(num, tail)
            num = float(num + '.' + tail[:2])
            tail = float('0.00'+ tail[2:])
            if num < 0:
                err -= tail
            elif num > 0:
                err += tail
            #print(err)
            if  (err >= 0.01) or (err <= -0.01):
                if num>0:
                    num += 0.01
                    err = err - 0.01            
                elif num < 0:
                    num -= 0.01
                    err = err + 0.01
            return(num,err)
        
        #stretch lenght 
        mag_x = self.magnitude/100*self.magnitude_percent_X
        #print('mag_x',mag_x)
        mag_y = self.magnitude/100*self.magnitude_percent_Y
        #print('mag_y',mag_y)
        
        #cycle_time = self.stretch_phase_time+self.hold_phase_time+self.recover_phase_time+self.rest_phase_time
        
        stretch_speed_x = mag_x/self.stretch_phase_time
        stretch_speed_y = mag_y/self.stretch_phase_time
        recover_speed_x = mag_x/self.recover_phase_time
        recover_speed_y = mag_y/self.recover_phase_time
        #saves end time
        full_time = self.result_time
        #saves continuous error
        
        
        for counta in range (0,self.cycle_count):
            #stretch phase            
            for t in range(1,self.stretch_phase_time+self.delta_time,self.delta_time):
                motor_X1, self.stretch_err_x1 = fix((stretch_speed_x*self.delta_time)/2, self.stretch_err_x1)
                motor_X2, self.stretch_err_x2 = fix((stretch_speed_x*self.delta_time)/2, self.stretch_err_x2)
                motor_Y1, self.stretch_err_y1 = fix((stretch_speed_y*self.delta_time)/2, self.stretch_err_y1)
                motor_Y2, self.stretch_err_y2 = fix((stretch_speed_y*self.delta_time)/2, self.stretch_err_y2)
                self.result[full_time + t] = ((motor_X1, motor_X2), (motor_Y1, motor_Y2))
                
            full_time = full_time + self.stretch_phase_time  
            #hold phase
            for t in range(1,self.hold_phase_time+self.delta_time,self.delta_time):
                self.result[full_time + t] = ((0,0),(0,0))
            full_time = full_time + self.hold_phase_time
            #recover phase    
            for t in range(1,self.recover_phase_time+self.delta_time,self.delta_time):
                
                motor_X1, self.recovery_err_x1 = fix((-recover_speed_x*self.delta_time)/2, self.recovery_err_x1)
                motor_X2, self.recovery_err_x2 = fix((-recover_speed_x*self.delta_time)/2, self.recovery_err_x2)
                motor_Y1, self.recovery_err_y1 = fix((-recover_speed_y*self.delta_time)/2, self.recovery_err_y1)
                motor_Y2, self.recovery_err_y2 = fix((-recover_speed_y*self.delta_time)/2, self.recovery_err_y2)
                self.result[full_time + t] = ((motor_X1, motor_X2), (motor_Y1, motor_Y2))
                
            full_time = full_time + self.recover_phase_time
            #rest phase                
            for t in range(1,self.rest_phase_time+self.delta_time,self.delta_time):
                self.result[full_time + t] = ((0,0),(0,0))   
                              
            full_time = full_time + self.rest_phase_time    
            self.result_time = full_time   