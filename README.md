# Improvements to *A Neural Algorithm of Artistic Style*

## University of Michigan EECS 445 Artistic Style

**Collaborators**: Brad Frost (@bfrost2893), Kevin Pitt (@kpittumich15), Nathan Sawicki, Stephen Kovacinski (@Kovacinski), Luke Simonson (@lukesimo)

This project was influenced by the paper [*A Neural Algorithm of Artistic Style*](http://arxiv.org/abs/1508.06576). This was the final project of
our EECS 445: Introduction to Machine Learning.

To use the script to make your own awesome artistic photos, you will need to have an NVIDIA GPU with CUDA drivers installed. This code would take hours upon hours when running on the CPU, but we may add this option if necessary.

### Installation

```bash
git clone https://github.com/bfrost2893/eecs445-project
pip install -r requirements.txt
```
Make sure that your machine has the CUDA drivers installed. If your machine is
configured to work with Theano GPU support, it will work as expected. See [this
documentation article](http://deeplearning.net/software/theano/tutorial/using_gpu.html)
for support on configuring Theano to work with NVIDIA GPUs.

### Usage
This script should work for most major Linux distributions. We implemented using
it Ubuntu 14.04 LTS (Trusty). However, it should work for any distribution.

Generate an image using `content/content.jpg` image and `style/picasso` paintings with
K = 4 nearest neighbors and weights 0.4, 0.3, 0.2, and 0.1 weights on the applied
style images.

```bash
python art.py --content content/content.jpg --style style/picasso -k 4 -w 0.4,0.3,0.2,0.1
```

Generate an image using `content/content.jpg` image and `style/picasso.jpg`, which
only applies one style image to the content with no other options.

```bash
python art.py --content content/content.jpg --style style/picasso.jpg
```

### Options

* `--content`, `-c`
    - The content image that you wish to apply style to
    - This must end with either '.jpg' or '.png'
* `--style`, `-s`
    - The style directory or image that you wish to apply
to the content image you specify
    - This must be a relative directory to the script and must not begin
    with '/', just the name. For example 'content/monet'
* `-k`
    - The number of style images you wish to apply
    - This will choose the best four style images according to the
    Euclidean distance between the content and the style images (a simple [k-Nearest Neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) implementation that we used for preprocessing)
* `--weights`, `-w`
    - Add custom weights to each neighbor image using a comma separated string
    - Example: `-w .6,.4`
    - The weights must sum to one and must equal the K value
