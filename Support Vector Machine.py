#   SVM is a binay classification algorithm which creates a hyper plane between the two datatypes.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

data_dict={-1:np.array([[1,7],
                       [2,8],
                       [3,8]]),
          1:np.array([[5,1],
                     [6,-1],
                     [7,3]])}

class support_vector_machine:
    def __init__(self,visualization=True):
        self.visualization=visualization
        self.colors={1:'r',-1:'b'}
        
        if(visualization):
            self.fig=plt.figure()
            self.ax=self.fig.add_subplot(1,1,1)
            
            
    def visualize(self):
        print(self.w,self.b)
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]
        
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v)/w[1]
        datarange=(self.min_feature_value*0.9,self.max_feature_value*1.1)
        hyp_x_min=datarange[0]
        hyp_x_max=datarange[1]
        
        psv1=hyperplane(hyp_x_min,self.w,self.b,1)
        psv2=hyperplane(hyp_x_max,self.w,self.b,1)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2],'k')
        
        nsv1=hyperplane(hyp_x_min,self.w,self.b,-1)
        nsv2=hyperplane(hyp_x_max,self.w,self.b,-1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2],'k')
        
        db1=hyperplane(hyp_x_min,self.w,self.b,0)
        db2=hyperplane(hyp_x_max,self.w,self.b,0)
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2],'y--')
        plt.show()
            
   #    We predict by pluging in features of the dataset into the equation of the hyperplane and take the sigh of 
    #   the answer as the catagory
    def predict(self,features):
        classification=np.sign(np.dot(self.w,np.array(features))+self.b)
        if(classification!=0) and self.visualization:
            self.ax.scatter(features[0],features[1],s=100,marker='*',
                            c=self.colors[classification])
        return classification
    
    #    In order to find the w and b for the hyper plane we use convex optimization.
    def fit(self,data):
        all_data=[]
        for i in data:
            for j in data[i]:
                for k in j:
                    all_data.append(k)
        
        min_value=min(all_data)
        max_value=max(all_data)
        
        self.min_feature_value=min_value
        self.max_feature_value=max_value
        
        transforms=[[1,1],
                  [-1,1],
                  [1,-1],
                  [-1,-1],]
        
        steps=[max_value*0.1,
              max_value*0.01,
              max_value*0.001]
        
        w_val=max_value*10
        b_step=5
        b_size=5
        
        all_data=None
        
        optimize_values={}

#   We start with smaller steps and then decrease them as we eleminate the region not containing the minima
        for step in steps:
            optimized=False
            #   Here we only consider values of w with equal magnitude in all dimensions for sake of reducing computation
            w=np.array([w_val,w_val])
            while not optimized:
                for b in np.arange((-1*b_size*max_value),(b_size*max_value),step*b_step):
                    for transform in transforms:
                        w_t=w*transform
                        check=True
                        for i in data:
                            for j in data[i]:
                                yi=i
                                if not yi*(np.dot(w_t,j)+b)>=1:
                                    check=False
                                    
                        if check:
                            optimize_values[np.linalg.norm(w_t)]=[w_t,b]
                        
                if(w[0]<0):
                    optimized=True
                    print('optimized')
                else:
                    w=w-step
            norm=sorted([n for n in optimize_values])
            selected=optimize_values[norm[0]]
            self.w=selected[0]
            self.b=selected[1]
            w_val=selected[0][0]+step*2
            
            
            
svm=support_vector_machine()
svm.fit(data=data_dict)
predict=[[0,10],[1,3],[3,4],[3,5],[5,5],[5,6],[6,-5],[5,8]]
for p in predict:
    svm.predict(p)
svm.visualize()  
                    
        
