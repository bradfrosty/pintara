# Improvements to *A Neural Algorithm of Artistic Style*

## University of Michigan EECS 445 Artistic Style

**Collaborators**: Brad Frost (@bfrost2893), Kevin Pitt (@kpittumich15), Nathan Sawicki, Stephen Kovacinski (@Kovacinski), Luke Simonson (@lukesimo)

This project was influenced by the paper [*A Neural Algorithm of Artistic Style*](http://arxiv.org/abs/1508.06576). This was the final project of
our EECS 445: Introduction to Machine Learning.

To use the script to make your own awesome artistic photos, you will need to have an NVIDIA GPU with CUDA drivers installed. This code would take hours upon hours when running on the CPU, but we may add this option if necessary.

Example usage:

```bash
python art.py --content content/content.jpg --style style/picasso -k 4
```

**Options**

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
