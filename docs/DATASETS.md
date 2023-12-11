# How to Install Datasets

`$DATA` denotes the location where datasets are installed, e.g.


[Out-of-distribution Generalization](#ood-generalization)
- [PACS](#pacs)
- [VLCS](#vlcs)
- [Office-Home-DG](#office-home-dg)
- [Digits-DG](#digits-dg)


## OOD Generalization

### PACS

Download link: [google drive](https://drive.google.com/open?id=1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE).

File structure:

```
pacs/
|–– images/
|–– splits/
```

You do not necessarily have to manually download this dataset. Once you run ``tools/train.py``, the code will detect if the dataset exists or not and automatically download the dataset to ``$DATA`` if missing. This also applies to VLCS, Office-Home-DG, and Digits-DG.

### VLCS

Download link: [google drive](https://drive.google.com/file/d/1r0WL5DDqKfSPp9E3tRENwHaXNs1olLZd/view?usp=sharing) (credit to https://github.com/fmcarlucci/JigenDG#vlcs)

File structure:

```
VLCS/
|–– CALTECH/
|–– LABELME/
|–– PASCAL/
|–– SUN/
```

### Office-Home-DG

Download link: [google drive](https://drive.google.com/open?id=1gkbf_KaxoBws-GWT3XIPZ7BnkqbAxIFa).

File structure:

```
office_home_dg/
|–– art/
|–– clipart/
|–– product/
|–– real_world/
```

### Digits-DG

Download link: [google driv](https://drive.google.com/open?id=15V7EsHfCcfbKgsDmzQKj_DfXt_XYp_P7).

File structure:

```
digits_dg/
|–– mnist/
|–– mnist_m/
|–– svhn/
|–– syn/
```