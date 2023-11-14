### -------2022年02月24日更新日志-------
1. 增加3D which的算法，算法可以兼容nii、nii.gz、nrrd格式的数据作为输入。
2. 增加生存模型，Cox回归、C-Index、列线图等。
3. 优化代码结构，增加案例的说明文档。
4. 更新crop_max_roi工具，增加对nii、nrrd的支持。

### -------2022年04月10日更新日志-------
1. 优化深度学习特征提取模块，增加深度学习特征教学视频。
2. 增加3DDenseNet模型（源代码模式）
3. 增加工具箱的教学视频。
4. crop_max_roi工具支持窗宽、床位设置。
5. 修复若干bug，优化用户体验。

### -------2022年04月16日更新日志-------
1. 增加What3D的3D模型特征提取模块
2. OKT-crop_max_roi新增crop出3d roi区域，并保存为nii.gz
   |- 使用`axis_3d`参数 > 4即可实现。
3. 新增调试通过的Which3D模块，并且支持批量Inference。
4. 解决Which组件，可视化过程中，不支持测试样本的bug。
5. 新增OKT-convert_jpg2nii，将jpg图像转化成nii的工具。
6. 修复其余若干bug，优化用户体验。

### -------2022年04月24日更新日志-------
1. 恢复之前误删的【What-概览模块】
2. 增加What3D模块，但目前仍存在于源码模式，需要在下次更新之后封装成低代码模式。
3. OKT-crop_max_roi新增crop出来非联通的独立区域，对于roi个数大于1的情况适用。
4. 将列线图转化成python版本，需要安装指定包。
    |- conda install r-base r-stringi r-rms
    |- pip install rpy2
    某些情况下需要配置R环境变量
    R_HOME          C:\OnekeyPlatform\onekey_envs\Lib\R
    R_USER          C:\OnekeyPlatform\onekey_envs\Lib\site-packages\rpy2

### -------2022年05月01日更新日志-------
1. 修正Onekey中传统组学提取多个label特征，出现特征覆盖的问题。
2. 修复了若干bug，改善了体验。
3. 在comp2中，基于comp1筛选之后进行深度学习升级的模块。
4. 修改了confusion matrix的计算bug。

### -------2022年05月08日更新日志-------
1. 修正which模块(segmentation)中由于标注label的overlap造成识别错误的问题。
   1. 增加了`labels`参数可以自行转化数据时指定overlap顺序。
2. Which算法中增加Dice指标计算。
3. 发布了列线图的OnekeyVideo教程，在baidu网盘下载。
4. 修正了一些OnekeyTools中的bug，如必要，可从baidu网盘下载更新。
5. 新增序列数据模块-Sequence，目前仍是源代码模式。
6. 修正了一些bug，改善了体验。

### -------2022年05月15日更新日志-------
1. OKT-convert_jpg2nii工具，修改文件输出目录不一致情况，兼容数值文件名排序。
2. 增加What2D-可视化模块，二维数据，使用Grand-CAM进行热图激活可视化。
3. 增加What3D-可视化模型，三维数据，使用Grand-CAM进行热图激活可视化。
4. 增加OKT-resample工具，可以根据输入的分辨率，进行采样，对于多中心，不同分辨率的数据统一到一个相同分辨率。
5. 所有的模块增加`transfer_learning`参数，支持全加载或者部分加载预训练模型参数。
6. 修正了一些bug，改善了体验。

### -------2022年05月22日更新日志-------
1. 修正可视化.bat脚本在某些情况下出现bug的情况。
2. 修正[WHAT-概览]中valid阶段图像尺寸出现bug的问题。
3. 增加DCA决策曲线。
4. 正式增加OnekeyComp-modules模块，Onekey模块化API教程
   1. ICC校验的模块。
5. 增加可视化模块的教程。
6. What3D升级到低代码模式。

### -------2022年05月29日更新日志-------
1. 新增OKT-fix_spacing工具，解决特征提取过程中由于数据设置空间不匹配，产生的【Image/Mask geometry mismatch】。
2. 新增OKT-rtstruct2nii工具，讲rtstruct数据转化成nii格式，要求：rtstruct数据与原始dicom数据在同一目录。
3. OKT-crop_max_roi新增对±n截取采样的支持.
   
    > 例如采样最大roi区域的±2和±4，OKT-crop_max_roi.exe --no_crop --surrounds 2 -2 4 -4  
4. fix OKT-crop_max_roi窗宽、窗位设置不生效的问题。
5. 在使用Lasso交叉验证进行特征筛选时，由于数据问题出现lambda=1,造成没有特征筛选出来的问题，增加强制筛选参数
   
    > `ensure_lastn`: 指定使用最后非增的倒数第几个lambda值。
6. DCA曲线增加多条曲线同时绘制功能。
7. 修正了一些bug，改善了体验。

### -------2022年06月05日更新日志-------
1. 新增OKT-crop_video_frame工具，抽取造影、内镜、超声视频图像帧。两种使用方法。
   > 1.默认抽取视频i帧。
   > 
   > 2.使用`seconds_per_frame`，可以自定义多少秒钟抽取一帧。
   
2. 增加了若干使用的教学视频。
   1. What、What模型可视化、数据标注等
   2. Which、数据标注、模型训练等
   3. ICC、列线图等。
3. 增加2D的What、Which相关模型的样例数据（更新至OnekeyDS中）
4. 增加源代码模式，`fusion`模块，可以使用Unet3D进行分割的同时进行分类模型。
   
    > 代码位置：onekey_core\models\fusion\unet_sc.py
5. comp8-Modules，新增【指标汇总】模块
   1. 新增低代码模式DCA曲线。
   2. ROC不同策略汇总结果。
   3. 新增DCA决策曲线汇总。（需要依赖Python3.7）
   4. Delong检测
6. 修正了一些bug，改善了体验。

### -------2022年06月12日更新日志-------
1. 传统组学模块新增多分类、回归任务组件。解决多分类、回归问题。
2. 修正OKT-convert2nii的spacing、origin之类参数的问题（使用原始参数）。
3. 新增OKT-gen_probably_map[Beta v0.1]，可以绘制病理切片的probably map以及prediction map。
   > 注意只能使用OKT-crop2path以及【What-概览】的valid目录一起使用。
4. 修正了若干bug，改善了体验。

### -------2022年06月19日更新日志-------
1. 结构化数据模块全部增加保存预测结果模块。
2. 新增stats数据统计模块，在Modules，主要用户临床数据统计，均值、方差、数量、占比以及p_value等。
3. 修正What3D中的roi_size参数问题。
4. OKT-resample模块采样之后，修改原始数据meta信息，例如spacing等。
5. OKT-crop2path工具，增加对OKT-gen_probably_map的功能支持。
6. OKT-gen_probably_map发布1.0正式版，可以可视化【What-概览】训练的val日志目录内容。 
7. 修正了若干bug，改善了体验。

### -------2022年06月26日更新日志-------
1. 在metrics增加ppv，npv，f1等指标支持。
2. 新增Metrics分析模块，可以对目前大多数的log日志进行指标分析。
3. 新增OKT-crop_WSI2patch工具，支持svs等病理切片数据直接进行原图crop。
4. 新增hosmer_lemeshow_test对校准曲线的评价函数。
5. 临床数据统计stats模块中，支持分label，细项进行统计。
6. 传统组学中新增多中心的组件，Comp1-传统组学-multicenter。

### -------2022年07月03日更新日志-------
1. 修正了一些OKT-crop_WSI2patch中的一些bug。
2. 所有的Onekey工具库进行更新，减小了工具的体积，并且test pass。
3. 新增2D分割中，Unet Family的源代码模式。
4. 新增OKT-standardlize工具，可以对所有的数据，去掉指定分位数之外的数值。
5. 修正了之前的一些bug，改善了用户体验。

### -------2022年07月10日更新日志-------
1. 修正了一些OKT-crop_WSI2patch中的一些bug。
2. What组件中新增Key2模块，用于将训练的模型转化成histogram或者tfidf特征。
3. 新增VAE的源代码模式，用于生信基因组降维或者2d图像重建。
4. 优化fusion源代码模式，增加dnn进行多组学特征融合。
5. 优化大多数包含图片的模块，可以保存SVG矢量图，对论文需要的素材更加友好。
6. 组学相关模块中的评价新增PPV、NPV、Precision、Recall、F1等指标。
7. 修正了之前的一些bug，改善了用户体验。

### -------2022年07月17日更新日志-------
1. 修正OKT-crop_WSI2patch由于图像过大，造成pixel too large的问题。
2. 新增smote重采样的源码模式，位于comp1.smote_resample。
3. 多因素逻辑回归模块，在【Comp8-Modules】。
4. 修正了一些lasso cv参数的问题。
5. Fix，Nomogram在生存分析模块的问题。
6. 修正了之前的一些bug，改善了用户体验。

### -------2022年07月14日更新日志-------
1. 修复了【Metric】模块F1值计算的问题以及其他指标的精度问题。
2. 新增【Comp1传统组学任务-ICC以及数据划分】，配合发布新的OnekeyDS。
    > 支持自定义数据划分。
   > 
    > 支持ICC校验联合后续验证。
3. What2D在模型训练的时候，保存最佳的训练Epoch参数，同时保存训练集预测结果。。
4. 新增【统计分析】，字符串直接map2numerical设置。
5. 修正了之前的一些bug，改善了用户体验。

### -------2022年07月31日更新日志-------
1. 修正「Metric」里面的【plt not found】错误。 
2. 新增What组件中，List模式下的交叉验证数据生成组件--【生成CV数据-List模式】
3. 修正了可视化模块中由于日志文件名变化造成的错误。
4. 修正了之前的一些bug，改善了用户体验。


### -------2022年08月07日更新日志-------
1. 解决当样本非常少时，【统计分析】模块中连续变量产生空行不对齐的问题.
2. 优化【What-概览】 

   >a. 增加最大准确率log，输出到模型训练最后。
   >
    >b. 在样本不均衡时，增加batch_balance参数，可以动态保证样本均衡。
3. 修正inception在inference的时候，加载出错的问题。
4. 新增What批量预测模块【What批量inference】。
5. 优化了【key2histogram】模块，当病理数据规模特别大时的执行效率问题。
6. 病理数据特别多时，增加测试集【val_max2use】参数，限制测试集数量。
7. 新增空值填充源代码，连续值填充均值，离散值填充中位数。
   > src: onekey_algo.custom.components.comp1.fillna
   > 
   > Usage: from onekey_algo.custom.components.comp1 import fillna
8. 修正了之前的一些bug，改善了用户体验。

### -------2022年08月14日更新日志-------
1. 新增【OKT-N4偏置场交验】工具。
2. 新增【OKT-mask_padding】工具，自动瘤内瘤周生成工具
    > 外padding，可以绘制Mask对应的瘤周数据
   > 
   > 向内padding，以绘制Mask对应的瘤内数据
   
3. 解决Which组件可视化无结果的bug。
4. 修正了之前的一些bug，改善了用户体验。

### -------2022年08月21日更新日志-------
1. 优化Comp2-生存率分析里面的AUC计算方法，并且增加DCA展示。
2. 优化Comp1-传统组学中样本特征可视化模块，增加特征统计模块，包括ttest、utest特征筛选、分布图。
    > 1.增加饼图展示特征占比
   > 
    > 2.增加箱型图展示ttest结果
3. 修正Comp2.1里面auc的计算口径问题。
4. 新增【OKT-convert_series2nii】组件，可以把系列的dcm文件转化成nii数据。
5. 【OKT-convert_jpg2nii】增加对扩展名大小写的兼容。
6. 优化所有工具的目录校验方式。
7. 修正了之前的一些bug，改善了用户体验。

### -------2022年08月28日更新日志-------
1. Which组件中，批量预测模块，增加Mask文件输出，修正保存Mask的尺度。
2. 新增Sol组件，以解决方案的方式，介绍Onekey的综合使用方法。
3. 修正【OKT-crop_max_roi】组件中roi_only参数在2维数据下不生效的问题。
4. 修正了之前的一些bug，改善了用户体验。

### -------2022年09月04日更新日志-------
1. 计算NPV、PPV之类的指标时，支持通过YouDen指数进行重算预测结果（默认采用）。
    > user_youden = False，采用的是argmax逻辑。
2. 解决极端情况下OKT-cropWSI2patch工具裁剪失败的问题。
3. 修正样本可视化时输出结果叠加，以及造成jupyter崩溃的问题。
4. 【OKT-gen_probably_map】增加predict_downsample参数，predict尺寸降采样。
5. 修正和新增若干【Which3D-概览】组件中的问题
    > 1.增加VNet模型支持。
   > 
   > 2.解决ResSegNet模型无法加载的问题。
   > 
   > 3.增加可以自定义Transform的功能。
   > 
   > 4.增加目前支持模型的说明，对使用更加友好。
   
6. 修正了之前的一些bug，改善了用户体验。

### -------2022年09月11日更新日志-------
1. 【OKT-standardize】增加窗宽窗位设置，同时支持对比度调整。
2. 【OKT-gen_probably_map】解决生存predict map时分辨率问题。
3. 针对多中心的数据，特征支持分组z-score功能。
4. 优化get_bst_split程序的metrics返回结果，可以通过模型，指定中位数之类的划分。
5. 调整了Comp1组件中的一些默认系数。
6. Comp8-Modules中增加【高级树模型】的一些特征重要性以及决策路径（需要自行安装graphviz）。
7. 修正了之前的一些bug，改善了用户体验。

### -------2022年09月18日更新日志-------
1. 修正【What-特征提取】组件，csv数据读取格式错误问题。
2. 发布Onekey-Solution模块，一键运行生成论文。
   1. 论文所有测试指标
   2. 论文所有的图、表
   3. 论文对应的英文描述初稿
3. 所有组件当中，删掉DecisionTree模型（一般效果最差）。
4. 修正了之前的一些bug，改善了用户体验。

### -------2022年09月25日更新日志-------
1. 解决ICC校验，数值为负数的情况。
2. 【OKT-mask_padding】工具，增加`keep_value`参数，可以保证某些参数在向外扩展的时候保持对应数据不变。
3. 【OKT-WSI2patch】工具增加多进程支持，可以并发处理多个文件（使用`num_prcess`参数指定，默认为1）。
4. 【OKT-crop_max_roi】工具，在指定`mask_id`时，通过roi_only只保留对应maskid的ROI区域。
5. 修正了之前的一些bug，改善了用户体验。

### -------2022年10月02日更新日志-------
1. 所有的统计指标，变更tp、tn等指标默认不使用youden index计算指标，保证与后文的混淆矩阵一致。
2. Comp9-sol1增加外围config配置，真正做到不用修改代码，只修改配置即可运行。
3. 修正What3D-特征提取组件的特征读取格式错误。
4. 【OKT-convert2nii】,针对多模态无法提取数据，增加筛选第一个模态功能。
5. Comp1以及Comp9的传统组学模块，增加样本预测直方图。
6. 改善特征统计检验模块中出现「cant not convert string to float」的兼容性。
7. 所有的组件，LASSO筛选都统一到一个数据集。
8. 增加【OKT-patch2predict】工具,通过颜色直方图的方式，筛选出那些patch[几乎]全白的patch不进行predict，以提高预测效率。
9. 修正【Which3D-概览】中【type object 'params' has attributed 'cached_ratio'】问题。
10. 修正了之前的一些bug，改善了用户体验。

### -------2022年10月09日更新日志-------
1. 所有传统组学Lasso特征筛选默认使用训练集，按需使用全部数据。
2. OKT-series2nii组件解决文件名特殊字符无法保存的问题。
3. 修正了之前的一些bug，改善了用户体验。

### -------2022年10月16日更新日志-------
1. 解决OKT-patches2predict、OKT-crop_WSI2patch组件在多进程中，出现无法运行的bug。
2. sol组件中，添加sample预测分布图。
3. 解决sol-config配置模式，参数解析问题，同时解决方案中增加config样例配置文件，不写代码，修改文件一键生成。
4. What组件中，所有的并行度都改成0（`j=0`），兼容配置比较差的机器。
5. 新增【sol2. 传统组学-多中心（指定数据集）-临床】，手动指定数据集划分（多中心任务）。
6. nomogram模块支持自定义保存文件名。
7. 修正了之前的一些bug，改善了用户体验。

### -------2022年10月22日更新日志-------
1. 增加NDR，IDI指标用于评价两个模型的好坏
   ```
   from onekey_algo.custom.components.metrics import NDR, IDI
   ```
2. comp8-单_多因素回归等组件增加说明文档。
3. 【OKT-crop_WSI2patch】增加`tif`，`mrxs`文件格式支持。
4. 修正Comp9-Sol1中，使用config配置的一些问题。
5. 优化了Comp9-Sol1和Sol2中间的一些默认参数配置，使使用变得更加丝滑。
6. fix了一些【OKT-convert_geojson2mask】工具的bug。
7. 修正了之前的一些bug，改善了用户体验。

### -------2022年10月30日更新日志-------
1. 优化「OKT-standardize」工具的参数检查以及功能提示。
2. 「OKT-resample」工具增加指定采样具体维度的功能， 
   > --new_res -1 -1 5，表示前两个维度不变，最后一个维度采样到相同的5mm大小。
3. 增加每个交叉验证结果的详细输出。
4. 「OKT-convert_jpg2nii」，单独输出的jpg图片也是一个3维结构。
5. 修正comp9中若干相关系数造成的bug。 
6. 修正了之前的一些bug，改善了用户体验。

### -------2022年11月06日更新日志-------
1. OKT-convert2jpg工具增加对tif数据的支持（tif不能太大，否则无法加载到内存）。
2. 针对生境分析，增加2个研究工具
   1. 「OKT-gen_sub」，基于两个模态数据进行差值计算。除了可以在生境分析使用，还可以在Delta Radiomics进行使用。
   2. 「OKT-gen_habitat_cluster」，基于Kmeans算法，对mask进行聚类，得到生境不同的类别。
3. 新增sol3-瘤内瘤周的解决方案【由于时间问题，仍在测试中，慎用！】。
4. 新增病理切片Patch归一化工具「OKT-patch_normalize」。
5. 修正了之前的一些bug，改善了用户体验。

### -------2022年11月13日更新日志-------
1. 特征筛选的CV增加ElasticNet的交叉验证筛选。
   > okcomp.comp1.lasso_cv_coefs(X_train, y_train, column_names=None, model_name='ElasticNet')
   
2. 调试完成Sol3的瘤内瘤周解决方案。
3. 修正「OKT-gen_habitat_cluster」的cluster指定个数不生效的问题。
4. 修正了之前的一些bug，改善了用户体验。

### -------2022年11月20日更新日志-------
1. Comp1增加Naive Bayes（朴素贝叶斯）、AdaBoost、GradientBoost模型支持。
   > ```model_names = [..., 'NB', 'AdaBoost', 'GradientBoosting']```
   > 
   >同时增加clf、reg模型的支持（NB只有分类模型支持）
   > 
2. 特征筛选模型，去掉跟label的相关性，使逻辑更加严谨。
3. 修改【OKT-gen_habitat_cluster】工具的cluster_id起始ID。
4. 优化【OKT-mask_padding】的计算逻辑，生成外扩区域速度更快。
5. 优化【OKT-convert_jpg2nii】大小写的支持。
6. 修正了之前的一些bug，改善了用户体验。

### -------2022年11月27日更新日志-------
1. Comp2结构化数据，新增4个组件。
   1. 结构化数据-二分类
   2. 结构化数据-回归
   3. 结构化数据-多中心二分类
   4. 结构化数据-多分类
2. 开放【OKT-crop_WSI2path】工具的level为负数的问题。
3. 修正Comp9-sol3的配置文件，config.txt, 瘤内瘤周ID的配置。
4. 修正了之前的一些bug，改善了用户体验。

### -------2022年12月04日更新日志-------
1. 修正特异性（specificity）和灵敏度（sensitivity）的计算口径。 
   > 将原来使用Yuden指数作为计算依据替换为0.5作为判断依据，如果还原，可使用`used_youden=True`
2. 增加Sensitivity，Specificity，NPV、PPV的95% CI计算。
   > [1] Wilson, E. B. "Probable Inference, the Law of Succession, and Statistical Inference," Journal of the American Statistical Association, 22, 209-212 (1927).
3. 【OKT-patch2predict】，增加verbose的debug参数，可以辅助调整参数。
4. 校准曲线增加平滑操作，`draw_calibration(..., smooth=True)`，如果遇到无法绘制问题，请关闭，默认不开启。
5. Fix极端情况下，What-List模式数据无法找到的问题。
6. 【OKT-crop_max_roi】通过`with_mask`保存，可以保存mask文件。
7. 【What-概览】组件增加对多通道输入的支持。
8. 修正了之前的一些bug，改善了用户体验。


### -------2022年12月11日更新日志-------
1. 【OKT-crop_WSI2patch】工具，增加level参数指定放大倍率，e.g. `--level 20x`则是使用20x的方法倍率（不一定完全匹配，会寻找最相近的放大倍率）。
2. Comp7-Survival新增【生存汇总Cox-KM-Nomo】模块，完整解决生存分析整体流程。
3. 新增多模态数据解决方案【Comp9-sol4.传统组学-多中心-多模态-临床】，多模态数据需要配合新的OnekeyDS。
   > 下载新的数据资源包：https://pan.baidu.com/s/1-OUvXs0xA0Af_Y4kBLZDPQ?pwd=ks97#list/path=%2F
   
4. fix 【Comp3-Ensemble融合】组件的一些指标不对齐的bug。
5. 修正了之前的一些bug，改善了用户体验。


### -------2022年12月18日更新日志-------
1. 特征筛选，增加mRMR功能。
   > 需要安装pymrmr：pip install pymrmr
   > ```python
   > from onekey_algo.custom.components.comp1 import select_feature_mrmr
   > data: pandas.DataFrame
   > select_feature_mrmr(data, num_features=10)
   > ```
2. 修复Comp2-量表数据，calc_sens_spec函数参数问题。
3. 修正了之前的一些bug，改善了用户体验。

### -------2022年12月25日更新日志-------
1. 优化【点我运行.bat】脚本的启动逻辑，可以在管理员权限下自动定位到onekey_comp。
2. 优化【OKT-crop_WSI2patch】中原始数据没有元信息造成无法裁剪的问题。
3. 优化【OKT-patch2predict】的并行处理逻辑，在极端情况下，依然可以高速并行处理。
4. 修正了之前的一些bug，改善了用户体验。

### -------2023年01月01日更新日志-------
1. 新的一年，新的开始。
2. 修正了之前的一些bug，改善了用户体验。

### -------2023年01月08日更新日志-------
1. 增加一些【点】和【线】的操作，可以求直线交点、垂足、距离等，一般用在骨科的Xray的任务较多。
2. Fix 【OKT-patch2predict】组件的del_image参数bug。
3. 解决3d分割模型UNETR在进行inference时模型参数设置的问题。
4. 修正了之前的一些bug，改善了用户体验。

### -------2023年01月15日更新日志-------
1. 优化What-key2组件中，概览的表示。
2. 树模型feature importance增加原始信息支持。
3. 修正了【OKT-WSI2patch】工具，在level找不到时出现bug，修正为使用最大level。
4. 所有的自定义提取特征增加【ngtdm】特征。
5. 在所有的解决方案中增加生存分析汇总分析。
6. 修正了之前的一些bug，改善了用户体验。

### -------2023年01月22日更新日志-------
1. 祝大家春节快乐，万事如意~~

### -------2023年01月29日更新日志-------
1. 增加【OKT-crop_WSI2patch】工具对qupath4.0+之后版本的支持。
2. mRMR的特征筛选功能，增加对非数值列的兼容。
3. 【What-3D概览】增加日志输出，向【What概览】的日志格式靠拢。
4. 【点我运行】启动项回复按运行位置启动，增加【点我运行-应急】文件，应对管理员启动时的启动位置。
5. 修正了之前的一些bug，改善了用户体验。

### -------2023年02月05日更新日志-------
1. 修正What3D模型训练过程中，输出的日志文件名与结果顺序不一致的Bug。
3. fix【OKT-convert2jpg】工具无法启动的问题。
4. 在统计分析map2numerical函数中，增加`map_nan`参数，是否映射空值，默认不进行映射，可以在后续任务进行填充。
5. 所有传统组学任务中，增加examplePET.yaml，用于提取PET数据的组学特征。
6. 所有OKT史诗级更新——增加欢迎截面。
7. 修正了之前的一些bug，改善了用户体验。

### -------2023年02月12日更新日志-------
1. 优化【What3D概览】以及【What3D-特征提取】组件的结构以及文字教程说明。
2. 在计算Sens、Spec这些参数的时候，默认使用Youden index进行统计，这个与混淆矩阵存在差异，混淆矩阵用的是0.5的概率阈值。
3. 优化【OKT-cropWSI2path】工具的level参数配置问题。
4. 修正了之前的一些bug，改善了用户体验。

### -------2023年02月19日更新日志-------
1. 【OKT-gen_habitat_cluster】增加--separate参数，可以每个样本单独生境区域生成。
2. 在所有的config.txt中新增3个画图的参数。所有任务下只要有这config.txt文件即可。
   1. figure.dpi：生成图像的DPI分辨率，默认300。
   2. figure.figsize：图像的大小，默认10*8。
   3. font.size：字体的大小，默认15。
3. 修正了之前的一些bug，改善了用户体验。


### -------2023年02月26日更新日志-------
1. 优化【OKT-crop_WSI2patch】中指定level放大倍数，但是存在奇偶性导致无法找到正确分辨率的问题。
2. 增加【OKT-rad_feature_extraction】，可以生成按体素的特征，方便后续生境分析使用。
3. 【OKT-gen_probably_map】工具新增生成数据增加alpha透明通道。
4. 【OKT-patch2predict】优化了“白”色背景的计算方法，默认使用新的算法。
5. What组件中，由于可能出现重名的情况，所有的日志输出数据的绝对目录。
6. config.txt，增加【display.precision】配置，可以全局指定保留的小数点位数。
7. 修正了之前的一些bug，改善了用户体验。


### -------2023年03月05日更新日志-------
1. 优化统计检验模块，默认所有的离散变量是卡方，二元连续变量是ttest，多元是ANOVA
2. 优化【单-多因素分析】模块，进行低代码模式优化，简单指定即可完成特征筛选。
3. 大幅度优化comp9中所有解决方案的【临床基线统计分析】组件。
   >自动识别部分去掉config.txt对应的配置即可，如果遇到不符合预期的自动识别，请继续使用config进行配置。
   1. 增加【单多因素分析】模块。
   2. 自动stats_column识别，不用指定分析特征列，自动识别。
   3. 自动mapping_columns列识别，临床特征可以不用进行数值化，Onekey自动完成映射。
   4. 自动continous_columns列识别，如果列不是整数，或者列的元素超过5个，则呗认定为连续特征。
   5. 自动缺失值填充，
4. 修正了之前的一些bug，改善了用户体验。

### -------2023年03月12日更新日志-------
1. 统一所有组学组件中特征分布于p_value分布的颜色图。
2. 组学特征提取支持并行特征提取，速度更快，使用`workers=n`进行并行提取特征。
3. 生成样本分布增加`threshold`参数，可以使用youden指数，指定划分点。
4. 新增【OKT-gen_roi_rad_features】，可以生成2D或者3D的全尺寸特征，同时可以可视化这些特征，尤其是纹理特征。
5. 优化【OKT-crop_video_frame】工具按照时间裁剪的参数配置。
6. 传统组学组件，修复特征提取时，某些样本存在特征提取失败，造成程序整体挂掉的问题。
7. 修正了之前的一些bug，改善了用户体验。

### -------2023年03月19日更新日志-------
1. fix多中心模块，样本可视化出错的问题。
2. fix 95% CI下限显示错误问题。
3. fix组件中默认workers太大（改成默认1个进行）造成崩溃的问题。
4. 新增comp9-sol5,深度学习解决方案
   1. 所有的配置可以在config.yaml文件进行参数配置。
   2. 后续所有配置都会想yaml格式迁移，过渡阶段可以同时在代码中配置。
   3. 这个解决方案，直接到Paper.md敬请期待~~。
5. 新增【What-效能评估】组件，可以评估深度学习的模型效能，目前此组件仅针对二分类模型。
6. 修正了之前的一些bug，改善了用户体验。

### -------2023年03月26日更新日志-------
1. 新增【OKT-patch2predict】的brightness_threshold可以指定删除的范围。
2. 【OKT-gen_habitat_cluster】生成的mask增加spacing的校验。
3. 解决【OKT-gen_roi_rad_features】工具指定param_file在多进程中不生效的问题。
4. 新增【Where概览】组件，支持目标检测算法。
5. Comp5整体改名为【Comp5-自动识别（Which, Where）】
6. 修正了之前的一些bug，改善了用户体验。

### -------2023年04月02日更新日志-------
1. 新发布目标检测练习数据集，需要OnekeyDS自行下载。
2. 【OKT-convert_series2nii】，增加多进程支持，加速处理进度。
3. 附件增加【example_3d_features.yaml】用于【OKT-gen_roi_rad_features】。
4. 修正了之前的一些bug，改善了用户体验。

### -------2023年04月09日更新日志-------
1. 增加[draw_predict_score]中threshold的参数类型支持。
2. 新增【comp9-sol6】病理弱监督学习解决方案。
   1. 单中心从Step0开始运行。多中心从Step1开始运行。
   2. 所有的配置文件迁移到【yaml】文件格式，支持注释。
3. 所有的Nomogram修改为600 DPI（高清图）
   > 所有Nomogram汇总的部分，需要修改width和height设置。
4. 所有的配置文件，增加对[DISABLE_VIDEO]的支持，可以手动不显示组件中的视频，减少资源占用。
5. 修正了之前的一些bug，改善了用户体验。

### -------2023年04月16日更新日志-------
1. 修正【单多因素回归】里面，多因素回归分析，保存日志错误问题。
2. 新增超分重建组件【comp0-Module1】。 需要同步更新2个资源文件。
   1. pretrain
   2. OnekeyVideo
3. 新增超分重建组件对饮的教学视频（已内嵌入组件）。
   - 更多关于超分重建算法：https://www.bilibili.com/video/BV1WX4y1z7tv/
4. 在资源文件中，新增【paper_class.rar】，汇总张老师看过的部分文章。
5. 修正了之前的一些bug，改善了用户体验。

### -------2023年04月23日更新日志-------
1. Lasso（EfficientNet）进行特征筛选的时候，支持Multi-Task.
2. 绘制校准曲线，支持remap参数，重新对结果进行一次映射。
3. 超分重建，增加对CPU的支持。
4. fix超分重建，内核挂掉的问题。
5. Survival Nomogram分析中，支持使用`x_range`参数指定x坐标范围。
6. 修正了之前的一些bug，改善了用户体验。

### -------2023年04月30日更新日志-------
1. 2D ROI自动分割，inference函数，增加post_process参数，自定义后处理方法。
   > input dir可以自己指定list集合。
2. 优化了【OKT-gen_probably_map】工具的显示色彩空间。
3. 修正了之前的一些bug，改善了用户体验。

### -------2023年05月07日更新日志-------
1. 【OKT-gen_habitat_cluster】增加了不同cluster下的Calinski-Harabasz分数。
2. 优化了单、多因素分析输出结果的表格表头。
3. 【OKT-resample】优化了工具的采样的方法， 改进之后会更加高效。
4. 新增【OKT-mask_super_resolution】工具，配合影像超分重建，可以生成mask对应mask的采样结果。
5. 修正了之前的一些bug，改善了用户体验。

### -------2023年05月14日更新日志-------
1. 修正Which组件【downsample_ratio】的问题。
2. fix统计检验模块，当存在val数据集时无法返回结果报错的问题。
3. 修复了影像检查配置模块的错误提示。
4. 新增【Comp1-Lasso-Cox】模块，可以直接进行生存回归。
   1. 新增get_x_y_survival，将数据划分成回归模型需要的数据格式。
   2. 新增get_prediction，获得cox模型的预测结果。
   3. 新增lasso_cox_cv，从lasso曲线趋，交叉验证图以及Weights、公式一体化输出。
5. 修正了之前的一些bug，改善了用户体验。

### -------2023年05月21日更新日志-------
1. 解决【OKT-convert_series2nii】工具由于一个模态造成全部数据解析失败退出的问题。
2. 新增【Which概览-ImageMask模式】，同时增加config配置文件。
   1. image，原始数据目录
   2. mask, mask文件存放的目录
   3. seg_labels.txt，每行一个label，表明mask中不同id的标签。
3. 修正了之前的一些bug，改善了用户体验。

### -------2023年05月29日更新日志-------
1. 优化【okt-update】更新工具。
2. 修正了之前的一些bug，改善了用户体验。

### -------2023年06月04日更新日志-------
1. 【OKT-crop_WSI2patch】工具支持标注中name指定。针对组织芯片确定患者所属区域。
2. 新增【OKT-convert_nii2jpg】可以裁剪所有的nii数据的所有的slice。
3. Which-Image_Mask模式，增加general_image_mask数据格式，可以兼容多分类的分割。
4. Fix 【OKT-patch2predict】中裁剪边缘数据的无法删除白色图片的问题。
5. 修正了之前的一些bug，改善了用户体验。

### -------2023年06月11日更新日志-------
1. Which-概览组件，增加Unet模型原生支持。
2. 【OKT-gen_roi_rad_features】使用功能float32，压缩一半的数据存储空间。
3. 修正了之前的一些bug，改善了用户体验。

### -------2023年06月18日更新日志-------
1. 修正了【OKT-gen_habitat_cluster】的数据类型不支持的问题。
2. fix【OKT-crop_video_frame】工具对空格文件的报错的问题。
3. 【OKT-gen_roi_rad_features】升级。
   1. 增加缓存机制，如果不指定overwrite，默认开启缓存，不用断电重跑。
   2. 增加skip_threshold，当体素的点超过阈值之后，跳过这个样本。
4. 【OKT-gen_habitat_cluster】去掉中间log，增加输出Calinski-Harabasz分值均值。
5. 【Which-概览】增加Unet支持 
6. 修正了之前的一些bug，改善了用户体验。

### -------2023年06月25日更新日志-------
1. 【OKT-gen_roi_rad_features】增加了对边缘数据的支持。
2. 修正了之前的一些bug，改善了用户体验。

### -------2023年07月02日更新日志-------
1. 在Comp1以及Comp9-Sol1和Sol2中，ROC汇总、DCA、样本分布、混淆矩阵增加训练集结果输出。
   > 其他模块，复制代码，参考实现即可。
2. 新增【Comp1-Lasso-Cox生存分析-多中心】，使用Lasso模型构建生存分析模块。 
3. 新增3D模型【ShuffleNet、DenseNet121, DenseNet121, DenseNet121, DenseNet121】
4. 单、多因素回归输出日志为OR值。
5. 解决config为空时，返回bug。 
6. 修正了之前的一些bug，改善了用户体验。

### -------2023年07月09日更新日志-------
1. 更新【Comp8-树模型可解释性】为【Comp8-模型可解释性】，进行模型可解释性分析。
   1. 逻辑回归，LR模型支持公式输出。
   2. 所有的树模型（RF、ER、XGBoost、LightGBM）支持特征重要性
   3. XGBoost、LightGBM支持决策树可视化。
2. 统计检验模块增加正太分布检查，自动根据是否是正太分布进行ttest或者utest。
3. 重新启用病理的2个工具，给jpg病理图像的同学一线生机。
   1. 【OKT-convert_geojson2mask】，将数据转化成mask格式。
   2. 【OKT-crop2patch】,将jpg数据重新转化。
4. 修复【What3D-特征提取】组件默认参数问题造成训练好的模型无法加载。
5. 修正了之前的一些bug，改善了用户体验。

### -------2023年07月16日更新日志-------
1. 修复【UNet】模型，当图像尺寸不是正方形时无法inference的问题。
2. 修复【What-3D特征提取】的参数问题。
3. 在特征提取的时候，增加post_process参数，可以对提取的特征进行后处理，例如avgpool等。
4. 发布了包含Covid19分割数据的数据
   1. 需要更新【OnekeyDS】
5. 优化传统组学进行数据检查的地方。
   1. 增加报错提示，【请检查你的配置，在{path}目录 ……】。
   2. 增加文件检查！
6. 【OKT-gen_habitat_cluster】组件增加ori_value，默认False，对特征进行标准化。
7. 纠正统计检验，正态分布结果分析的列名。
8. 创建分类模型，增加【create_clf_model_none_overfit】，所有树模型不容易过拟合。
9. 更新【shapiro_results】由于版本问题造成无法解析的问题。
10. 修正了之前的一些bug，改善了用户体验。

### -------2023年07月23日更新日志-------
1. 【OKT-gen_habitat_cluster】解决由于原始数据是int，造成为0的问题。
2. 增加所有组件获取数据的兼容性，忽略掉扩展名之后的完全匹配。
3. 所有的分类任务，默认增加数据【exists】检查，所有病理任务，删除数据之后，不需要再手动删除record。 
4. 优化数据检查存在逻辑，可以支持子目录。
5. 增加【create_clf_model_none_overfit】，在所有初始化这些模型的地方，修改使用这个none_overfit即可。
   ```python
   models = okcomp.comp1.create_clf_model(model_names)
   models = okcomp.comp1.create_clf_model_none_overfit(model_names)
   ```
6. 修正了之前的一些bug，改善了用户体验。

### -------2023年07月30日更新日志-------
1. 【OKT-gen_probably_map】更新
   1. 病理的热图，默认增加cmap图条。
   2. 优化热图的展示区域，让概率为0的区域更加精准。
2. Lasso-Cox相关的底层调用，并行度调为None，兼容CPU不好的情况。
3. 优化NRI，IDI指标，确认结果可用。
4. 优化【OKT-gen_roi_rad_features】的数据量跳过机制。
5. 【What-可视化.bat】修正多个模型可视化时出现的图像异常。
6. Fix bug in http://medai.icu/thread/662.
7. 修正了之前的一些bug，改善了用户体验。

### -------2023年08月06日更新日志-------
1. config中新增【lines.linewidth】参数，可以选择画图的线宽。
2. 更新【OKT-crop_video_frame】功能，增加递归指定目录。
3. Onekey更新到3.0.0，增加众创功能。
    > 更新最新资源文件：10篇论文带你走完科研的前世今生.zip
   >  链接：https://pan.baidu.com/s/16xgwGkBhW6ubzNj7gJ0Zxg?pwd=cmiw 
    --来自百度网盘超级会员V7的分享
4. 修正了之前的一些bug，改善了用户体验。


### -------2023年08月13日更新日志-------
1. 【OKT-crop_WSI2patch】,增加strategy参数，默认为None，可选quality或者speed。
   > 速度优先：取最接近匹配的分辨率的较低的。
   > 质量优先：取最接近匹配的分辨率的较高的。
   > None：当找不到匹配的分辨率时，使用最高分辨率。

2. 优化get_param_in_cwd的获取方式。
3. 所有的任务的配置文件升级到YMAL格式
   > 具体Ymal语法：http://medai.icu/thread/677
   
4. 新增【OKT-convert_each_channel_2nii】工具，针对核磁不同b值或者存在多个通达的数据的情况。
5. 新增【Comp9-Sol7】，多模态，多中心组件。
6. 修正了之前的一些bug，改善了用户体验。

### -------2023年08月20日更新日志-------
1. 优化【Comp8-指标汇总】组件，可以通过指定路径进行任意Onekey输出的结果进行汇总。
2. 优化【Comp9-Sol3】解决方法，兼容目前所有主流的瘤内、瘤周方案，同时支持任意结果汇总。
   > 优化Nomogram的显示逻辑，可以使用临床特征进行绘制Nomogram。
   > 优化所有的汇总逻辑，delong检测结果。
   
3. 修正了之前的一些bug，改善了用户体验。

### -------2023年08月27日更新日志-------
1. 修正【Comp9-Sol7多模态】组件，影像融合时没有区分训练集的问题。
2. 修复了极端情况下多线程提取组学特征，无法提取的问题。
3. 修复【OKT-gen_sub】
   1. 工具在样本数据类型为int是无法保存的问题。
   2. norm参数不生效的问题。
4. 【What-概览】，【What3D-概览】增加输出每个样本预测概率的功能，以csv格式输出到日志。
5. 【What3D-特征提取】新增【init_from_onekey3d】，可以直接使用3D模型训练路径，不需要额外配置参数。
6. 新增【OKT-convert_svg4paper】，将svg图像转化成png、tif、pdf等期刊要求的任意格式。
7. 增加【OKT-crop_max_roi】的兼容性。
8. 修正了之前的一些bug，改善了用户体验。

### -------2023年09月03日更新日志-------
1. 【OKT-crop_WSI2patch】更新
   1. keep_largest_one参数在极端情况下，mask与image分辨率不匹配的问题。
   2. 当存在mask_save_dir参数时。可以把patch对应的裁剪的mask一起输出。
2. 【OKT-convert2nii】，增加对极端情况下的nrrd格式数据支持。
3. 【Comp9-Sol7】 fix bug。
4. 【OKT-gen_roi_rad_features】，默认使用use_3d参数，如果想使用2d，进行特征可视化，可以用use_2d参数！
5. 修正了之前的一些bug，改善了用户体验。

### -------2023年09月10日更新日志-------
1. 将多分类算法进行抽象，融合到【draw_roc_per_classes】功能。
   1. 多分类自动可以输出metric
   2. 新增【get_binary_gt】，【get_binary_prediction】功能，可以直接多分类，生成one vs others预测数据。
2. fix 工具【OKT-convert_each_channel2nii】元信息丢失问题。
3. 优化2D 分割算法，关于数据集需要去images和masks交集的问题。
4. 【OKT-crop_WSI2path】，增加mask_save_dir功能，根据保存的数据，可以进行2D分割。
5. 修正了之前的一些bug，改善了用户体验。

### -------2023年09月17日更新日志-------
1. 增加【OKT-convert_serises2nii】工具对中文描述的。
2. 增减了Comp9-sol8，深度学习解决方案。
   1. 样本进行伪多中心划分。
   2. 深度学习模型训练。
3. 修正了之前的一些bug，改善了用户体验。

### -------2023年09月24日更新日志-------
1. 新增【OKT-convert_kfb2svs】，将所有的kfb文件转成svs。
2. 修复【Comp9-sol3】在使用mrmr时基于全部样本进行筛选的问题。
3. 改造【OKT-standardize】，增加参数校验以及非同时生效警告。
4. 修正了之前的一些bug，改善了用户体验。

### -------2023年10月01日更新日志-------
1. 祝大家节日快乐，祝祖国繁荣昌盛。


### -------2023年10月15日更新日志-------
1. 基于深度学习的分类模型，追加ViT的模型支持。
   1. ViT，refer：https://openreview.net/pdf?id=YicbFdNTTy
   2. SimpleViT。 An update from some of the same authors of the original paper proposes simplifications to ViT that allows it to train faster and better.
   3. Onekey v3.1.4之前的版本，需要升级运行时环境，双击【安装依赖.bat】即可；链接：https://pan.baidu.com/s/1qCFeDytpqtt5TuEExG_8eQ?pwd=onek
2. 【OKT-crop2path】功能合并入【OKT-cropjpg2path】。 
3. 解决【OKT-gen_probably_map】对编码方式不支持的问题。
4. 更新【What-病理切片】组件中的list、folder功能。
5. 修正了之前的一些bug，改善了用户体验。

### -------2023年10月22日更新日志-------
1. 增加Comp8增加【human_boost】、【生成伪多中心数据】组件。
   1. 【human_boost】对比结果具体数据形式，参考data/human.csv
   2. 【生成伪多中心数据】，可以生成伪多中心数据。使用这种数据，结果会更加严谨。
2. 优化了【Comp8-指标汇总】组件的汇总逻辑，可以参考data目录下的数据形式进行汇总。
3. 为了保证与混淆矩阵一致，将计算指标地方use_youden默认的True修改为False。
4. 将所有的单中心lasso，如果使用共`get_bst_split`, 默认修改为使用全部数据进行。
5. Fix 【OKT-crop_max_roi】工具中，裁剪2D数据存在的问题。
6. 修正了之前的一些bug，改善了用户体验。

### -------2023年10月29日更新日志-------
1. 解决统计分析中，由于样本少，造成显示错误的问题。
2. 修改多中心组件中=='val'为!='train'
3. fix 【What-3D概览】folder模式的model_config没有定义的问题。
4. 优化Which3D-概览的模型描述。
5. 修正了之前的一些bug，改善了用户体验。

### -------2023年11月05日更新日志-------
1. Fix 【OKT-convert2nii】不能转化nii数据的问题。
2. 【What3d-模型可视化】，使用init_from_onekey3d接口。
3. 解决生境分析最后一公里的特征融合问题。 针对所有样本放在一起聚类，支持3种不同的特征融合方法
   1. 特征合并
      1. max, 所有具有相同特征名称的生境区域，保留最大值。
      2. min, 所有具有相同特征名称的生境区域，保留最小值。
      3. mean, 所有具有相同特征名称的生境区域，保留均值。
   2. fill,基于KNN方法，对所有的缺失值进行填充，与MICE方法类似。
   3. remap，根据一个锚点，根据相似度进行计算，重新映射这些生境区域。
4. fix一个目标检测中的bug。
5. 修正了之前的一些bug，改善了用户体验。