# This project is derived from Udacity's mini-project "Sentiment Analysis Network":
# https://github.com/udacity/deep-learning-v2-pytorch/tree/master/sentiment-analysis-network

import time
import sys
import numpy as np

from collections import Counter

# Encapsulate our neural network in a class
class SentimentNetwork:


    def __init__(self, reviews, labels, hidden_nodes = 10, min_count=20, polarity_cutoff=0.05, learning_rate = 0.1):
        """
        Create a SentimenNetwork with the given settings.

        Args:
            reviews(list) - List of reviews used for training
            labels(list) - List of POSITIVE/NEGATIVE labels associated with the given reviews
            hidden_nodes(int) - Number of nodes to create in the hidden layer
            min_count(int) - Words are only added to the vocabulary if they occur in the vocabulary more than min_count times.
            polarity_cutoff(float) - Words are only added to the vocabulary if the absolute value of their postive-to-negative ratio is at least polarity_cutoff.
            learning_rate(float) - Learning rate to use while training
        
        """
        # Assign a seed to our random number generator to ensure we get
        # reproducable results during development 
        np.random.seed(1)

        # Save parameters 
        self.min_count = min_count
        self.polarity_cutoff = polarity_cutoff

        # process the reviews and their associated labels so that everything
        # is ready for training
        self.pre_process_data(reviews, labels)
        
        # Build the network to have the number of hidden nodes and the learning rate that
        # were passed into this initializer. Make the same number of input nodes as
        # there are vocabulary words and create a single output node.
        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels):
        
        # Calculate the positive-to-negative ratios of words used in the reviews.

        positive_counter = Counter()
        negative_counter = Counter()
        total_counter = Counter()
        for label, review in zip(labels, reviews):
            words = review.split(' ')
            if label=='POSITIVE':
                positive_counter.update(words)
            else:
                negative_counter.update(words)
            total_counter.update(words)

        pn_ratios = Counter()
        for word, usage in total_counter.items():
            pn_ratios[word] = np.log(positive_counter[word]/(negative_counter[word] + 1.0))
            #if usage >= self.min_count: # drop rare words
            #    ratio = np.log(positive_counter[word]/(negative_counter[word] + 1.0))
            #    if ratio >= self.polarity_cutoff:
            #        pn_ratios[word] = ratio


        review_vocab = set()
        
        # populate review_vocab with all of the words in the given reviews
        for review in reviews:
            for word in review.split(' '):
                if total_counter[word] >= self.min_count and np.abs(pn_ratios[word]) >= self.polarity_cutoff:
                    review_vocab.add(word)
        #print(review_vocab)
        
        #test_review = 'ghost of dragstrip hollow is a typical      s teens in turmoil movie . it is not a horror or science fiction movie . plot concerns a group of teens who are about to get kicked out of their  hot rod  club because they cannot meet the rent . once kicked out  they decide to try an old haunted house . the only saving grace for the film is that the  ghost   paul blaisdell in the she creature suit  turns out to be an out of work movie monster played by blaisdell .  '
        #for word in test_review.split(' '):
        #    print(total_counter[word], " ", pn_ratios[word])


        # Convert the vocabulary set to a list so we can access words via indices
        self.review_vocab = list(review_vocab)
        
        label_vocab = set()
        for label in labels:
            label_vocab.add(label)
        
        # Convert the label vocabulary set to a list so we can access labels via indices
        self.label_vocab = list(label_vocab)
        
        # Store the sizes of the review and label vocabularies.
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
        
        # Create a dictionary of words in the vocabulary mapped to index positions
        self.word2index = {}
        # populate self.word2index with indices for all the words in self.review_vocab
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i
        
        # Create a dictionary of labels mapped to index positions
        self.label2index = {}
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i
         
        
    # Modify init_network:
    # You no longer need a separate input layer, so remove any mention of self.layer_0
    # You will be dealing with the old hidden layer more directly, so create self.layer_1, a two-dimensional matrix with 
    # shape 1 x hidden_nodes, with all values initialized to zero
    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Store the number of nodes in input, hidden, and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Store the learning rate
        self.learning_rate = learning_rate

        # Initialize weights
        
        # TODO: initialize self.weights_0_1 as a matrix of zeros. These are the weights between
        #       the input layer and the hidden layer.
        self.weights_0_1 = np.zeros(shape=(self.input_nodes, self.hidden_nodes))
        
        # TODO: initialize self.weights_1_2 as a matrix of random values. 
        #       These are the weights between the hidden layer and the output layer.
        self.weights_1_2 = np.random.rand(self.hidden_nodes, self.output_nodes)
        
        # layer_0 is no longer needed according to the instructions above
        #self.layer_0 = np.zeros((1,input_nodes))
        # create self.layer_1, a two-dimensional matrix with shape 1 x hidden_nodes, with all values initialized to zero
        self.layer_1 = np.zeros(shape=(1, self.hidden_nodes))
    
        
                
    def get_target_for_label(self,label):
        # TODO: Copy the code you wrote for get_target_for_label 
        #       earlier in this notebook. 
        return 1 if label=='POSITIVE' else 0
        
    def sigmoid(self,x):
        # TODO: Return the result of calculating the sigmoid activation function
        #       shown in the lectures
        return 1/(1+np.exp(-x))
    
    def sigmoid_output_2_derivative(self,output):
        # TODO: Return the derivative of the sigmoid activation function, 
        #       where "output" is the original output from the sigmoid fucntion 
        return output*(1-output)

    # Modify train:
    # Change the name of the input parameter training_reviews to training_reviews_raw. This will help with the next step.
    # At the beginning of the function, you'll want to preprocess your reviews to convert them to a list of indices 
    # (from word2index) that are actually used in the review. This is equivalent to what you saw in the video when Andrew 
    # set specific indices to 1. Your code should create a local list variable named training_reviews that should contain 
    # a list for each review in training_reviews_raw. Those lists should contain the indices for words found in the review.
    # Remove call to update_input_layer
    # Use self's layer_1 instead of a local layer_1 object.
    # In the forward pass, replace the code that updates layer_1 with new logic that only adds the weights for the indices 
    # used in the review.
    # When updating weights_0_1, only update the individual weights that were used in the forward pass.
    def train(self, training_reviews_raw, training_labels):
        
        # Create a local list variable named training_reviews that should contain a list for each review in training_reviews_raw. 
        # Those lists should contain the indices for words found in the review.
        training_reviews = []
        for review in training_reviews_raw:
            training_reviews.append(list(set([self.word2index[word] for word in review.split(' ') if word in self.word2index])))
            
        print("training reviews:", training_reviews)
        
        # make sure out we have a matching number of reviews and labels
        assert(len(training_reviews) == len(training_labels))
        
        # Keep track of correct predictions to display accuracy during training 
        correct_so_far = 0
        
        # Remember when we started for printing time statistics
        start = time.time()

        # loop through all the given reviews and run a forward and backward pass,
        # updating weights for every item
        for i in range(len(training_reviews)):
            
            # call to update_input_layer removed according to the instructions
            #self.update_input_layer(training_reviews[i])
            y = self.get_target_for_label(training_labels[i])
            
            # The forward pass through the network. 
            # That means use the given review to update the input layer, 
            # then calculate values for the hidden layer,
            # and finally calculate the output layer.
            #          
            # Do not use an activation function for the hidden layer,
            # but use the sigmoid activation function for the output layer.
            rev = np.array(training_reviews[i])
            if rev.size == 0:
                print("\nReview is empty! Consider descreasing polarity cutoff.\n")
                continue
            
            # my optimized implementation for sparse inputs of zeros and ones
            self.layer_1 = np.sum(self.weights_0_1[rev, :], axis=0, keepdims=True)
            #print("layer_1:", self.layer_1.shape)
            #hidden_layer = np.matmul(self.layer_0, self.weights_0_1) # (n_ex, n_input) x (n_input, n_hidden) => (n_ex, n_hidden)
            output_layer = np.matmul(self.layer_1, self.weights_1_2) # (n_ex, n_hidden) x (n_hidden, n_output) => (n_ex, n_output)
            output_layer = self.sigmoid(output_layer)
            
            # backpropagation
            output_error = y - output_layer # () - (n_ex, n_output) => (n_ex, n_output)
            output_error_term = output_error * self.sigmoid_output_2_derivative(output_layer)
            
            # (n_ex, n_output) x (n_output, n_hidden) => (n_ex, n_hidden)
            hidden_error = np.matmul(output_error_term, self.weights_1_2.T)
            hidden_error_term = hidden_error * 1 # no activation function is used for the hidden layer
            
            # (n_hidden, n_ex) x (n_ex, n_output) => (n_hidden, n_output)
            #delta_weights_1_2 = np.matmul(hidden_layer.T, output_error_term)
            delta_weights_1_2 = np.matmul(self.layer_1.T, output_error_term)

            #delta_weights_0_1 = np.matmul(self.layer_0.T, hidden_error_term)
            # no need to calculate all delta weights, since we are going to modify only a small part of them
            # as long as input values are only 0 and 1, we can avoid data duplication and use broadcasting to update weights_0_1
            delta_weights_0_1 = hidden_error_term
            
            self.weights_1_2 += self.learning_rate * delta_weights_1_2 / 1
            # When updating weights_0_1, only update the individual weights that were used in the forward pass.
            self.weights_0_1[rev,:] += self.learning_rate * delta_weights_0_1 / 1
            
            # Keep track of correct predictions. To determine if the prediction was
            # correct, check that the absolute value of the output error 
            # is less than 0.5. If so, add one to the correct_so_far count.
            if np.abs(output_error) < 0.5:
                correct_so_far += 1
            
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the training process. 

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")
    
    def test(self, testing_reviews, testing_labels):
        """
        Attempts to predict the labels for the given testing_reviews,
        and uses the test_labels to calculate the accuracy of those predictions.
        """
        
        # keep track of how many correct predictions we make
        correct = 0

        # we'll time how many predictions per second we make
        start = time.time()

        # Loop through each of the given reviews and call run to predict
        # its label. 
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1
            
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the prediction process. 

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i+1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
    
    # Modify run:
    # Remove call to update_input_layer
    # Use self's layer_1 instead of a local layer_1 object.
    # Much like you did in train, you will need to pre-process the review so you can work with word indices, 
    # then update layer_1 by adding weights for the indices used in the review.
    def run(self, review):
        """
        Returns a POSITIVE or NEGATIVE prediction for the given review.
        """
        # Run a forward pass through the network, like you did in the
        # "train" function. That means use the given review to 
        # update the input layer, then calculate values for the hidden layer,
        # and finally calculate the output layer.
        #
        #       Note: The review passed into this function for prediction 
        #             might come from anywhere, so you should convert it 
        #             to lower case prior to using it.
        #self.update_input_layer(review.lower())
        
        rev = list(set([self.word2index[word] for word in review.split(' ') if word in self.word2index]))
        rev = np.array(rev)
        if (rev.size == 0):
            print("\nReview is empty! Consider descreasing polarity cutoff.\n")
            return 'UNKNOWN'

        #hidden_layer = np.matmul(self.layer_0, self.weights_0_1) # (n_ex, n_input) x (n_input, n_hidden) => (n_ex, n_hidden)
        self.layer_1 = np.sum(self.weights_0_1[rev, :], axis=0, keepdims=True)
        #print("layer_1:", self.layer_1.shape)
        #hidden_layer = np.matmul(self.layer_0, self.weights_0_1) # (n_ex, n_input) x (n_input, n_hidden) => (n_ex, n_hidden)
        output_layer = np.matmul(self.layer_1, self.weights_1_2) # (n_ex, n_hidden) x (n_hidden, n_output) => (n_ex, n_output)
        output_layer = self.sigmoid(output_layer)
        
        
        # The output layer should now contain a prediction. 
        # Return `POSITIVE` for predictions greater-than-or-equal-to `0.5`, 
        # and `NEGATIVE` otherwise.
        return 'POSITIVE' if output_layer >= 0.5 else 'NEGATIVE'



def pretty_print_review_and_label(i):
    print(labels[i] + "\t:\t" + reviews[i][:80] + "...")

g = open('reviews.txt','r') # What we know!
reviews = list(map(lambda x:x[:-1],g.readlines()))
g.close()

g = open('labels.txt','r') # What we WANT to know!
labels = list(map(lambda x:x[:-1].upper(),g.readlines()))
g.close()

print("labels.txt \t : \t reviews.txt\n")
pretty_print_review_and_label(2137)
pretty_print_review_and_label(12816)
pretty_print_review_and_label(6267)
pretty_print_review_and_label(21934)
pretty_print_review_and_label(5297)
pretty_print_review_and_label(4998)

#mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.1)
#mlp.train(reviews[:-1000],labels[:-1000])

mlp = SentimentNetwork(reviews[:-1000],labels[:-1000],min_count=20,polarity_cutoff=0.8,learning_rate=0.01)
mlp.train(reviews[:-1000],labels[:-1000])
mlp.test(reviews[-1000:],labels[-1000:])