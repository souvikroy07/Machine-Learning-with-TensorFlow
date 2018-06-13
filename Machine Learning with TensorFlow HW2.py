
# Name : Souvik Roy


#Import libraries

import tensorflow as tf
import numpy as np

#Define input scope

with tf.name_scope('Input_Scope') as scope:

    #Define input placeholder
    a = tf.placeholder(tf.float32)


#Define Middle Scope


with tf.name_scope('Middle_Scope') as scope:  

    # Define nodes in the middle section
    b = tf.reduce_prod(a,name="b")
    c= tf.reduce_mean(a,name="c")
    d= tf.reduce_sum(a,name="d")
    e=tf.add(c,b,name="e")


#Define Outer Scope


with tf.name_scope('Output_Scope') as scope: 

    #Define Final Node
    f=tf.multiply(e,d)


#Start a new session

with tf.Session() as sess:

    # Define a random array of 100 elements with mean 1 and SD 2 
    rand_array = np.random.normal(1, 2, 100)

    # Execute the node
    print(sess.run(f, feed_dict={a: rand_array})) 

    #Get a GraphDef Object
    sess.graph.as_graph_def()

    #Write the graph to a file
    file_writer=tf.summary.FileWriter('./',sess.graph)

    #Close the writer
    file_writer.close()

    #Close the session
    sess.close()

