# MTAGN
The PyTorch implementation for multi-task attention guided network (MTAGN) in [End to End Multi-task learning with Attention for Multi-objective Fault Diagnosis under Small Sample](https://www.sciencedirect.com/science/article/abs/pii/S0278612521002521)
## Abstract
In recent years, deep learning (DL) based intelligent fault diagnosis method has been widely applied in the field of equipment fault diagnosis. However, most of the existing methods are mainly proposed for a single diagnosis objective, namely, they can only handle a single task such as recognizing different fault types (or locations) or identifying different fault severities. Besides, the scarce of data is a difficult issue because very few data could be obtained when a fault occurs. To overcome these challenges, a novel multi-task attention guided network (MTAGN) is proposed for multi-objective fault diagnosis under small sample in this paper. MTAGN consists of a task-shared network to learn a global feature pool and M task-specific attention networks to solve different tasks. With attention module, each task-specific network is able to extract useful features from task-shared network. Through multi-task learning, multiple tasks are trained simultaneously and the useful knowledge learned by each task could be utilized by each other to improve the performance. An adaptive weighting method is used in the training stage of MTAGN to balance between tasks and for better convergence results. We evaluated our method through three bearing datasets and the experimental results demonstrate the effectiveness and adaptability in different situations. Comparison experiment with other methods is also conducted in the same setup and the results proved the superiority of the proposed method under small sample.
## Citation
@article{xie2022end,
  title={End to end multi-task learning with attention for multi-objective fault diagnosis under small sample},
  author={Xie, Zongliang and Chen, Jinglong and Feng, Yong and Zhang, Kaiyu and Zhou, Zitong},
  journal={Journal of Manufacturing Systems},
  volume={62},
  pages={301--316},
  year={2022},
  publisher={Elsevier}
}
