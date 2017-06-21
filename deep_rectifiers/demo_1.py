#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 13:50:22 2017

@author: aidanrocke
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 13:29:51 2017

@author: aidanrocke
"""
import sys

sys.path.insert(0,'/path/to/mod_directory')

## let's define an elementary sorting algorithm:
    
def sort(x):
    n = len(x)
    if min(x[1:n-1]-x[0:n-2]) < 0:
        for passnum in range(len(x)-1,0,-1):
            for i in range(passnum):
                if x[i]>x[i+1]:
                    temp = x[i]
                    x[i] = x[i+1]
                    x[i+1] = temp
    return x

## figuring out whether a list is sorted:
    

#dataset 1: integers in the interval (0,100):
X_train1, Y_train1, X_test1, Y_test1 = datasets(0,100,10000,10,1)

#dataset 2: integers in the interval (-100,100):
X_train2, Y_train2, X_test2, Y_test2 = datasets(-100,100,10000,10,1)

#dataset 3: decimals in the interval (-1,1):
X_train3, Y_train3, X_test3, Y_test3 = datasets(-1,1,10000,10,0.5)

#dataset 4: decimals in the interval (-100,100):
X_train4, Y_train4, X_test4, Y_test4 = datasets(-100,100,10000,10,0.5)
    
   




#get the weights:
W = []

for layer in model.layers:
    W.append(layer.get_weights())

# in theory 12 hidden units should be enough:

#training on the first dataset:

history = model.fit(X_train1, Y_train1, nb_epoch=20, batch_size=500, verbose=1)

evaluation1 = model.evaluate(X_test1, Y_test1, verbose=1)

#check generalization:
    
# positive and negative integers
evaluation2 = model.evaluate(X_test2, Y_test2, verbose=1)
#97%

evaluation3 = model.evaluate(X_test3, Y_test3, verbose=1)
# 50% 

evaluation4 = model.evaluate(X_test4, Y_test4, verbose=1) 
# 89%

#when I don't use dropout with (10,12,1):
    
#evaluation1 = 99%

#evaluation2 = 97%

#evaluation3 = 50%

#evaluation4 = 89%

#when I don't use dropout with (10,1000,1):

#evaluation1 = 99.4%

#evaluation2 = 86%

#evaluation3 = 50%

#evaluation4 = 86%

#when I use dropout with (10,1000,1):
    
#evaluation1 = 99.6%

#evaluation2 = 99.48%

#evaluation3 = 50%

#evaluation4 = 99.4%






#if we do things in the other direction however we have:
    
# hardest:
history = model.fit(X_train3/100, Y_train3, nb_epoch=20, batch_size=500, verbose=1)

evaluation = model.evaluate(X_test3/100, Y_test3, verbose=1)

# hard:

evaluation = model.evaluate(X_test3, Y_test3, verbose=1)
#94%

evaluation = model.evaluate(X_test3/1000, Y_test3, verbose=1)
#50%





#training on the second dataset:
    
history = model.fit(X_train2, Y_train2, nb_epoch=20, batch_size=500, verbose=1)

evaluation = model.evaluate(X_test2, Y_test2, verbose=1)

#[0.0045329083524644374, 0.99939999999999996]

# training on the third dataset:
    
history = model.fit(X_train3, Y_train3, nb_epoch=20, batch_size=500, verbose=1)

evaluation = model.evaluate(X_test3, Y_test3, verbose=1)

# [0.23958186652064323, 0.90259999999999996]

# the fourth data set:
    
history = model.fit(X_train4, Y_train4, nb_epoch=20, batch_size=500, verbose=1)

evaluation = model.evaluate(X_test4, Y_test4, verbose=1)   

# [0.0045512530386447904, 0.99955000000000005]

# somehow the previous examples serve as prior knowledge of some kind and 
# helps us learn from new datasets but does this translate to better handling
# of dataset shift?

# can I be certain that my model has learned anything?

X_train5, Y_train5, X_test5, Y_test5 = datasets(3,4,10000,10,0.5)


evaluation = model.evaluate(X_test5, Y_test5, verbose=1)   

# [3.303561432647705, 0.5]

# let's go back to square 1:

# let's check the performance for random integer vs. random decimal sequences:
    


# why am I stuck at 50% accuracy for 'small' intervals?
# ss this due to a defect in my model or a weakness of implementation. 
    


# what if we assume no prior knowledge?


# now let's check to see whether this generalizes to other data well:

# positive and negative numbers:

    
evaluation = model.evaluate(X_test_2, Y_test_2, verbose=1)

#[0.044199950504908339, 0.98740000000000006]

# ok not bad, now let's try non-integer numbers as well:
X_unsorted = np.random.uniform(low=-1,high=1,size=(10000,10))
X_sorted = np.sort(X_unsorted,axis=1)

X_train = np.vstack((X_unsorted,X_sorted))
Y_train = np.vstack((np.zeros((10000,1)),np.ones((10000,1)))) 

X_test1 = np.random.randint(low=-1,high=1,size=(10000,10))
X_test2 = np.sort(X_test1,axis=1)

X_test = np.vstack((X_test1,X_test2))
Y_test = np.vstack((np.zeros((10000,1)),np.ones((10000,1))))

# permute both train and test data:
permutation = np.random.permutation(range(len(X_train)))
X_train, Y_train = X_train[permutation], Y_train[permutation]
X_test_3, Y_test_3 = X_test[permutation], Y_test[permutation]
    
evaluation = model.evaluate(X_test_3, Y_test_3, verbose=1)


### let's try things in the other direction now. 
### train on fractions in the interval (-1,1) and see how it generalizes:
    

# 50% accuracy

#going back to the integers: [0.60815023725601347, 0.95874999999999999]

#going to positive and negative integers:
#evaluation = model.evaluate(X_test, Y_test, verbose=1)
#96%


