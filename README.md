# Official Implementation of DeepSpeech Distances proposed in [*High Fidelity Speech Synthesis with Adversarial Networks*](https://arxiv.org/abs/1909.11646)

This repo provides a code for estimation of **DeepSpeech Distances**, new evaluation metrics for neural speech synthesis.

### **Details**

The computation involves estimating Fréchet and Kernel distances between high-level features of the reference and the examined samples extracted from hidden representation of [NVIDIA's DeepSpeech2](https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/deepspeech2.html) speech recognition model.

We propose four distances:


*   *Fréchet DeepSpeech Distance* (*FDSD*, based on FID, see [2])
*   *Kernel DeepSpeech Distance* (*KDSD*, based on KID, see [3])
*   *conditional Fréchet DeepSpeech Distance* (*cFDSD*),
*   *conditional Kernel DeepSpeech Distance* (*cKDSD*).

The conditional distances compare samples with the same conditioning (e.g. text) and asses conditional quality of the audio. The uncoditional ones compare random samples from two distributions and asses general quality of audio. For more details, see [1].

### **Usage**

To use the demo, [open the provided notebook in colab](https://colab.research.google.com/github/mbinkowski/DeepSpeechDistances/blob/master/deep_speech_distances.ipynb).

Alternatively, [open a new colab notebook](https://colab.research.google.com/), mount a drive and clone this repository:

```
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
!git clone https://github.com/mbinkowski/DeepSpeechDistances `/content/drive/My Drive/DeepSpeechDistances`
```
After that, go to */content/drive/My Drive/DeepSpeechDistances*, open a demo notebook *deep_speech_distances.ipynb*, and follow the instructions therein.

### **Notes**
We provide a tensorflow meta graph file for DeepSpeech2 based on the original one available with the [checkpoint](https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/deepspeech2.html). The provided file differs from the original only in the lack of map-reduce ops defined by horovod library; therefore the resulting model is equivalent to the original.

This is an 'alpha' version of the API; although fully functional it will be heavily updated and simplified soon.

### **References**

[1] Mikołaj Bińkowski, Jeff Donahue, Sander Dieleman, Aidan Clark, Erich Elsen, Norman Casagrande, Luis C. Cobo, Karen Simonyan, [*High Fidelity Speech Synthesis with Adversarial Networks*](https://arxiv.org/abs/1909.11646), ICLR 2020.

[2] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter, [*GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*](https://arxiv.org/abs/1706.08500), NeurIPS 2017.

[3] Mikołaj Bińkowski, Dougal J. Sutherland, Michael Arbel, Arthur Gretton, [*Demystifying MMD GANs*](https://arxiv.org/abs/1801.01401), ICLR 2018.
