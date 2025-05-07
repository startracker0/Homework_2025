# 《人工智能原理：模型与算法》作业
## 1.黑白棋(./reversi)
这是一个基于蒙特卡洛树搜索（MCTS）的黑白棋人机对战游戏。玩家可以选择黑棋或白棋，与AI进行对战。AI通过蒙特卡洛树搜索算法进行决策。由于我服务器没有配置可视化界面，故此工程下玩家需要通过输入坐标来下棋。
## 2.花卉分类任务报告(./flower)
本项目是一个基于 ResNet 的花卉图像分类任务。通过对花卉数据集的训练与测试，实现对多种花卉种类的准确识别和分类。

### 项目结构
```bash
./flower/
├── flower_data/         
│   ├── train/           
│   ├── val/             
│   └── test/            
├── outputs/             
│   ├── best_model_resnet50.pth 
│   ├── class_mapping.json      
│   ├── training_history.png    
│   └── predictions.png         
├── models/         
│   ├── resnet.py          
├── train.py             
├── test.py            
└── README.md           
```