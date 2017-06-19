# tv_script_generator
Requires Changes

3 SPECIFICATIONS REQUIRE CHANGES

Excellent implementation of your network! :blush: However, your model didn't fully converge yet and the training loss is > 1.0. I've added some suggestions to improve your code further.

You only need to make a few small changes to your code so I'm confident that you will pass this project with your next submission.
Required Files and Tests

The project submission contains the project notebook, called “dlnd_tv_script_generation.ipynb”.
All the unit tests in project have passed.
Perfect, you've passed all unit tests! :+1:
Preprocessing

The function create_lookup_tables create two dictionaries:

Dictionary to go from the words to an id, we'll call vocab_to_int
Dictionary to go from the id to word, we'll call int_to_vocab
The function create_lookup_tables return these dictionaries in the a tuple (vocab_to_int, int_to_vocab)
Nice job creating both dictionaries.

Suggestion: alternatively you can directly enumerate for the int_to_vocab:

def create_lookup_tables(text):  
    vocab = set(text)
    vocab_to_int = {c: i for i, c in enumerate(vocab)}
    int_to_vocab = dict(enumerate(vocab))

    return vocab_to_int, int_to_vocab
The function token_lookup returns a dict that can correctly tokenizes the provided symbols.
Excellent, you've successfully created a dict for all symbols.

Suggestion: I would advise you to be more consistent in the used values. For example, you're using mark and Mark (even though the examples in the notebook are also not consistent).
Build the Neural Network

Implemented the get_inputs function to create TF Placeholders for the Neural Network with the following placeholders:

Input text placeholder named "input" using the TF Placeholder name parameter.
Targets placeholder
Learning Rate placeholder
The get_inputs function return the placeholders in the following the tuple (Input, Targets, LearingRate)
Perfect, the get_inputs function creates the correct placeholders.
The get_init_cell function does the following:

Stacks one or more BasicLSTMCells in a MultiRNNCell using the RNN size rnn_size.
Initializes Cell State using the MultiRNNCell's zero_state function
The name "initial_state" is applied to the initial state.
The get_init_cell function return the cell and initial state in the following tuple (Cell, InitialState)
Nice job stacking multiple BasicLSTMCells. While stacking multiple cells will benefit your model performance in most cases, with the limited dataset we use in this project 3 stacked BasicLSTMCells is too complex. I would advise you to use 2 layers at maximum and you will notice that your network starts converging faster. Also, see my comments below.
The function get_embed applies embedding to input_data and returns embedded sequence.
:+1:
The function build_rnn does the following:

Builds the RNN using the tf.nn.dynamic_rnn.
Applies the name "final_state" to the final state.
Returns the outputs and final_state state in the following tuple (Outputs, FinalState)
Excellent job creating a function that builds the RNN!
The build_nn function does the following in order:

Apply embedding to input_data using get_embed function.
Build RNN using cell using build_rnn function.
Apply a fully connected layer with a linear activation and vocab_size as the number of outputs.
Return the logits and final state in the following tuple (Logits, FinalState)
Great!
The get_batches function create batches of input and targets using int_text. The batches should be a Numpy array of tuples. Each tuple is (batch of input, batch of target).

The first element in the tuple is a single batch of input with the shape [batch size, sequence length]
The second element in the tuple is a single batch of targets with the shape [batch size, sequence length]
Spot on!

Suggestion: another option without for loops and with a zip function would look like this:

def get_batches(int_text, batch_size, seq_length):
    n_batches = int(len(int_text) / (batch_size * seq_length))
    inputs = np.array(int_text[: n_batches * batch_size * seq_length])
    outputs = np.array(int_text[1: n_batches * batch_size * seq_length + 1])

    x = np.split(inputs.reshape(batch_size, -1), n_batches, 1)
    y = np.split(outputs.reshape(batch_size, -1), n_batches, 1)

    return np.array(list(zip(x, y)))
Neural Network Training

Enough epochs to get near a minimum in the training loss, no real upper limit on this. Just need to make sure the training loss is low and not improving much with more training.
Batch size is large enough to train efficiently, but small enough to fit the data in memory. No real “best” value here, depends on GPU memory usually.
Size of the RNN cells (number of units in the hidden layers) is large enough to fit the data well. Again, no real “best” value.
The sequence length (seq_length) here should be about the size of the length of sentences you want to generate. Should match the structure of the data.
The learning rate shouldn’t be too large because the training algorithm won’t converge. But needs to be large enough that training doesn’t take forever.
Set show_every_n_batches to the number of batches the neural network should print progress.
Nice start tuning your hyperparameters :blush: However, your model didn't fully converge yet. After decreasing the number of stacked BasicLSTMCells, I would advise you to tweak your hyperparameters again. Especially pay attention to:

The rnn_size represents the number of hidden units.
If the learning rate is too large, the algorithm won't fully converge. A smaller learning rate can help your model to avoid local minima but it can take longer to converge, try to find the right balance.
A bigger batch size can help your model to train more efficiently, as long as it fits in your memory.
The number of epochs should result in a low training loss and the model should be fully converged (not decreasing anymore and certainly not increasing again).
Suggestion: did you experiment with slightly shorter sequence length (seq_length) as well? For example, the average sentence length in the training data.
Also, the training loss is currently shown irregularly. If you want to output this more constantly you should use a value for show_every_n_batches that is dividable by the number of batches ( = #training_examples / batch_size).
The project gets a loss less than 1.0
Your final training loss is > 1.0. For this project we require a training loss < 1.0. You may have noticed that your model didn't fully converge yet after 80 epochs. To make your model converge faster I would advise you to stack 2 BasicLSTMCells instead of 3. If you add more training data, you'll probably benefit from more layers but the data is limited in our case.
Generate TV Script

"input:0", "initial_state:0", "final_state:0", and "probs:0" are all returned by get_tensor_by_name, in that order, and in a tuple
Correct.
The pick_word function predicts the next word correctly.
Excellent job adding some randomness to your function!

Suggestion: you can also use this oneliner to add randomness directly:
return np.random.choice(list(int_to_vocab.values()), 1, p=probabilities)
The generated script looks similar to the TV script in the dataset.

It doesn’t have to be grammatically correct or make sense.
As expected the TV script doesn't make too much sense. Lets re-evaluate your output after making the suggested changes.
