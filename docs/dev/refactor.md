# 本文件说明了当前存在的问题和改进方针

## 现有问题

### 配置文件混乱

配置文件的混乱很大程度也是因为判断逻辑欠整合
bot.py中的配置供scheduler中的预处理逻辑和handler中的回复使用
agent又使用专门的配置

### 判定逻辑混乱

理论上agent只用于ai判断,而本地逻辑判断应当作为预处理部分在入队前完成
现有的任务队列正是为此设计.一旦处理时间长短有极大差异的本地与agent任务混合投入,队列将可能面对一系列设计之外的调用场景,这可能严重影响性能(如:一个耗时长的agent任务卡住大量耗时短的本地判断任务,导致任务队列的容量被浪费)

## 解决方针

由处理机制重构带动配置文件重构

### 处理机制重构

#### 被放弃的方案

发展至此,本地预处理的复杂程度已经超出我一开始的预期,现在需要对scheduler进行拆分,分成纯粹的任务队列定义和实例新的scheduler.py和preprocessor.py 所有的预处理任务都会由preprocessor.py完成 processmsg等函数也将移入preprocessor.py

为了统一配置文件并将本地识别放入预处理任务,现有的agent也将被拆分,其中的所有本地判断都会被拆分放入preprocessor.py中然后保留agent调用的部分在原文件中.新的配置文件将被preprocessor和agent共用
为了将预处理和agent调用解耦,task类可能需要重构用来容纳指定任务所用的更多信息.

#### 目前新方案

从实际落地复杂度出发 现在看来scheduler并不能完全解耦前期判断和agent处理,这是由于前期处理和agent处理结合得过于紧密,例如agent判断是否需要回应 这就很难加入到这个模型中,因此我提出了一种新得架构能够高效整合这些更能同时尽可能少的重构

新的架构不按照流程分类,而使按照功能分类,文件也按照功能来分,同现有的summary.py auto_reply.py 一样,这样最大限度的保证了不同任务处理流程的独立.同时为了保证任务热点agent调用能被有序,高效的利用 将由所有的summary统一调用scheduler中的方法 await队列返回.这相当于将真正的核心agent调用使用scheduler进行了封装,提供了本地的agent池,同时scheduler的task也可以简化 只提供输入和输出就可以了

新的架构概述:summary reply等任务在 main.py中独立启动 然后独立运行 只在调用agent并获得reply这一步阻塞地调用scheduler的方法 从而实现同步

现在我将提供一个新的scheduler的demo 命名为agent_pool.py 以供参考
统一调用ai的核心函数(黑盒子 传入api url 提示词 传回文本) 然后给我 我会把agent_pool.py还有main.py中的启动agent_pool搞定
然后你这样用就可以了
当然输入参数还可以加
跟我说 这个我来实现
    reply = await agentp_LLM(
        api_key="sk-xxx",
        prompt="什么是量子计算？",
        priority=0,          # 高优先级插队
        temperature=0.3
    )
