window.siteContent = {
  author: "MomenT",
  posts: [
    {
      slug: "what-is-generative-model",
      title: "生成模型入门：从概率分布到分布拟合",
      date: "2026-05-22",
      category: "学习笔记",
      summary:
        "一篇真正从零开始的生成模型基础笔记。我们先不讨论 Diffusion、VAE 或 GAN，而是把随机变量、概率分布、总体与样本、似然、最大似然估计、KL 散度和分布拟合这些概念讲清楚。",
      tags: [
        "Generative Model",
        "Probability",
        "Maximum Likelihood",
        "KL Divergence",
        "Machine Learning"
      ],
      cover: "cover-c",
      body: String.raw`
        <p class="lead">在学习 Diffusion、VAE、GAN、Flow Matching 之前，最容易被跳过的问题是：生成模型到底是什么？它为什么总是在说“学习数据分布”？为什么训练目标会变成最大似然、KL 散度或者某种重建损失？这篇文章专门补这个地基。本文参考《深度学习入门5：生成模型》中步骤 1、步骤 2 和步骤 5 的讲法，但不会照搬书中的段落，而是按我们后续学习生成模型需要的逻辑重新组织。</p>

        <section class="article-section">
          <h2>0. 先把一句话说清楚</h2>
          <p>生成模型的核心目标可以先写成一句话：</p>
          <div class="insight-box">生成模型试图学习真实数据背后的概率分布，并利用这个学到的分布生成新的、像真实数据一样的样本。</div>
          <p>这句话里有三个关键词：数据、概率分布、生成。初学者最容易把注意力放在“生成”上，比如生成图片、生成文字、生成音乐。但理论上更根本的是前两个词：数据来自哪里？概率分布是什么？模型为什么可以通过学习分布来生成新样本？</p>
          <p>如果这些问题没想清楚，后面学习 Diffusion 时就会觉得很突然：为什么一张图片要看成高维随机变量？为什么说真实图片服从某个 \(p_{\mathrm{data}}(x)\)？为什么模型要学习 \(p_\theta(x)\)？为什么从高斯噪声开始采样可以得到图片？这些问题的共同根源，都是生成模型的基础概念。</p>
          <p>所以本文先不讲酷炫模型，也先不加可视化。我们只做一件事：把“生成模型是分布建模”这件事讲透。</p>
        </section>

        <section class="article-section">
          <h2>1. 数据不是孤立的点，而是从某个机制中来的</h2>
          <p>假设我们手里有一组身高数据：</p>
          <div class="equation">\[
            171.2,\quad 168.5,\quad 175.0,\quad 180.3,\quad 169.7,\quad \ldots
          \]</div>
          <p>这些数字看起来只是一个表格。但从统计建模的视角看，每个数字都不是凭空出现的，而是从某个总体中抽出来的样本。这个总体可以是“某年龄段男性的身高总体”，也可以是“某学校所有学生的身高总体”。</p>
          <p>总体通常很大，甚至我们永远无法完整观察。我们能拿到的只是有限样本。统计学和机器学习常常做的事情，就是用有限样本反推总体的性质。生成模型也是如此：我们看到有限训练数据，希望学到背后的数据生成机制。</p>

          <p>书里用“总体”和“样本”来解释这个关系。可以把它想成一条链：总体是真实世界中全部对象，采样让我们得到有限样本，参数估计再用有限样本反推总体规律。这个视角非常重要。训练集不是世界本身，训练集只是世界的一小部分观测。模型不能只是记住训练样本，而应该抓住训练样本背后的规律。只有这样，它才能生成训练集中没有的新样本。</p>
          <p>把这个想法换成机器学习语言，就是：</p>
          <div class="equation">\[
            x^{(1)},x^{(2)},\ldots,x^{(N)}
            \sim
            p_{\mathrm{data}}(x).
          \]</div>
          <p>这里 \(x^{(i)}\) 表示第 \(i\) 个观测样本，\(p_{\mathrm{data}}(x)\) 表示真实但未知的数据分布。我们不知道这个分布的完整数学式，只知道训练集像是从它里面抽出来的。</p>
          <p>这一步的心理转变很关键：数据集不是一堆死数据，而是某个未知分布给我们的线索。生成模型要做的，就是利用这些线索还原分布的形状。</p>
        </section>

        <section class="article-section">
          <h2>2. 随机变量和概率分布：生成模型的基本语言</h2>
          <p>为了谈分布，我们先需要“随机变量”。随机变量不是说变量本身很神秘，而是说它的取值由随机机制决定。掷骰子的点数是随机变量，人的身高也可以看成随机变量，一张图片也可以看成随机变量。</p>
          <p>如果随机变量是离散的，比如骰子点数，那么概率分布可以写成每个取值对应的概率：</p>
          <div class="equation">\[
            P(X=1)=\frac{1}{6},\quad
            P(X=2)=\frac{1}{6},\quad
            \ldots,\quad
            P(X=6)=\frac{1}{6}.
          \]</div>
          <p>如果随机变量是连续的，比如身高，单点概率通常没有意义。我们关心的是某个区间内的概率，比如身高在 170 到 175 厘米之间的概率。连续型分布用概率密度函数 \(p(x)\) 描述，区间概率通过积分得到：</p>
          <div class="equation">\[
            P(a\leq X\leq b)=\int_a^b p(x)\,dx.
          \]</div>
          <p>这意味着 \(p(x)\) 本身不是“取值正好等于 \(x\) 的概率”，而是概率密度。真正的概率来自面积，也就是积分。很多初学者第一次看到概率密度会困惑：为什么 \(p(x)\) 可以大于 1？原因就在这里。密度不是概率，面积才是概率。</p>
          <p>生成模型经常处理高维数据。图片可以看成一个向量：</p>
          <div class="equation">\[
            x\in\mathbb{R}^{H\times W\times C}.
          \]</div>
          <p>如果是 \(256\times256\) 的 RGB 图片，那么维度就是 \(256\times256\times3\)。此时 \(p_{\mathrm{data}}(x)\) 是一个极高维空间中的概率分布。这个分布非常复杂：绝大多数随机像素组合都不是自然图片，真正像图片的点只占据空间中很小、很有结构的一部分。</p>
          <p>所以，生成图像并不是“随机拼像素”，而是从一个高度结构化的分布中采样。能否生成像样的图片，取决于模型有没有学到这个分布的结构。</p>
        </section>

        <section class="article-section">
          <h2>3. 什么叫“对分布建模”？</h2>
          <p>真实分布 \(p_{\mathrm{data}}(x)\) 通常未知。我们只能选择一个模型族，用参数控制它的形状。把这个模型分布记作：</p>
          <div class="equation">\[
            p_\theta(x).
          \]</div>
          <p>这里 \(\theta\) 表示模型参数。对于一维正态分布，参数是均值和标准差：</p>
          <div class="equation">\[
            \theta=(\mu,\sigma).
          \]</div>
          <p>此时模型分布是：</p>
          <div class="equation">\[
            p_\theta(x)=
            \frac{1}{\sqrt{2\pi}\sigma}
            \exp\left(
            -\frac{(x-\mu)^2}{2\sigma^2}
            \right).
          \]</div>
          <p>如果我们用正态分布建模身高，那么“建模”就包含两个动作。</p>
          <ol>
            <li>选择模型形式：假设身高可以近似看成正态分布。</li>
            <li>估计模型参数：根据样本决定 \(\mu\) 和 \(\sigma\) 应该是多少。</li>
          </ol>

          <p>书中把这个过程放在身高数据上讲。先观察身高直方图，发现它大致像钟形曲线；于是选择正态分布作为模型；再用样本估计均值和标准差。这个例子虽然简单，但它已经包含生成模型的基本骨架：左边是我们选择的概率模型形式，右边是观测到的样本，中间的参数估计负责把样本信息转化为模型参数。</p>
          <p>更复杂的模型也是同样逻辑。GMM 用多个高斯分布叠加来拟合多峰数据；VAE 用神经网络和潜变量来表达复杂分布；Diffusion 用逐步加噪和反向去噪来学习数据分布。模型形式越来越复杂，但根本问题没变：我们要构造一个 \(p_\theta(x)\)，让它接近 \(p_{\mathrm{data}}(x)\)。</p>
        </section>

        <section class="article-section">
          <h2>4. 生成模型为什么要拟合分布？</h2>
          <p>现在回答最核心的问题：为什么生成模型的目标是拟合分布？</p>
          <p>因为如果模型分布 \(p_\theta(x)\) 足够接近真实分布 \(p_{\mathrm{data}}(x)\)，那么从 \(p_\theta(x)\) 采样出来的样本，就会像真实数据一样。也就是说：</p>
          <div class="equation">\[
            p_\theta(x)\approx p_{\mathrm{data}}(x)
            \quad\Longrightarrow\quad
            x_{\mathrm{new}}\sim p_\theta(x)
            \text{ looks like real data.}
          \]</div>
          <p>这句话把“分布拟合”和“生成新数据”连接起来了。我们不是直接规定每个新样本长什么样，而是先学会什么样的样本概率高，什么样的样本概率低。然后通过采样，就能得到新的合理样本。</p>
          <p>以身高为例，如果模型学到的是均值 172.7、标准差 4.8 的正态分布，那么从这个正态分布采样，得到 173.9、168.2、181.0 这样的数字都很合理。它们不一定出现在原训练集中，但符合身高总体的统计特征。</p>
          <p>以图像为例，真实图片分布会给“有清晰边缘、合理纹理、物体结构正确”的图片较高概率，给随机噪声图片极低概率。如果模型学到了这个分布，那么采样时就更可能得到自然图片，而不是花屏。</p>
          <p>因此，生成模型的“生成能力”来自“分布拟合能力”。拟合分布不是额外的数学包装，而是生成的本质。</p>
        </section>

        <section class="article-section">
          <h2>4.5 训练集如何代表真实分布？</h2>
          <p>这里还有一个细节必须讲清楚：我们说要拟合 \(p_{\mathrm{data}}(x)\)，但训练时根本拿不到这个真实分布的解析式。我们只有训练集。那模型到底是怎么“接触”真实分布的？</p>
          <p>答案是：训练集提供了对真实分布的采样近似。假设样本是独立同分布抽样得到的，那么对于任意函数 \(f(x)\)，真实分布下的期望是：</p>
          <div class="equation">\[
            \mathbb{E}_{p_{\mathrm{data}}}[f(x)]
            =
            \int p_{\mathrm{data}}(x)f(x)\,dx.
          \]</div>
          <p>由于不知道 \(p_{\mathrm{data}}\)，我们用样本平均近似这个期望：</p>
          <div class="equation">\[
            \mathbb{E}_{p_{\mathrm{data}}}[f(x)]
            \approx
            \frac{1}{N}\sum_{i=1}^{N}f(x^{(i)}).
          \]</div>
          <p>这就是蒙特卡洛近似的基本思想：用从分布中抽到的样本，近似分布上的积分。书中在讨论 KL 散度与最大似然关系时，也用到了这个想法。</p>
          <p>如果把训练集本身看成一个经验分布，可以写成：</p>
          <div class="equation">\[
            \hat{p}_{\mathrm{data}}(x)
            =
            \frac{1}{N}
            \sum_{i=1}^{N}
            \delta(x-x^{(i)}).
          \]</div>
          <p>这里的 \(\delta\) 可以理解为“把概率质量放在每个训练样本上的尖峰”。经验分布只在训练样本处有质量，而真实分布应该更平滑、更广泛。训练生成模型时，我们不能只死记这个经验分布，而要用模型族 \(p_\theta(x)\) 学出一个能概括总体规律的分布。</p>
          <p>这也解释了泛化问题。训练集越小，经验分布越粗糙，模型越容易误判总体形状；训练集越大，样本平均越能代表真实分布，参数估计越可靠。但即使训练集很大，模型族如果选得太弱，也仍然学不好。例如用一个单峰正态分布去拟合多种动物图片，模型表达能力显然不够。</p>
          <div class="insight-box">训练集不是目标本身，而是通向真实分布的窗口。生成模型真正想学的是窗口背后的世界。</div>
        </section>

        <section class="article-section">
          <h2>4.6 生成模型和判别模型有什么区别？</h2>
          <p>很多机器学习入门内容会先讲分类模型，比如判断一张图是猫还是狗。这类模型通常关心条件概率：</p>
          <div class="equation">\[
            p(y|x).
          \]</div>
          <p>它回答的问题是：给定输入 \(x\)，标签 \(y\) 是什么？这类模型常被称为判别模型，因为它直接学习如何区分类别边界。</p>
          <p>生成模型关心的问题不同。它更关心数据本身是如何出现的。最朴素的生成模型学习：</p>
          <div class="equation">\[
            p(x).
          \]</div>
          <p>如果是有条件生成，比如给定文字生成图片，则学习：</p>
          <div class="equation">\[
            p(x|c),
          \]</div>
          <p>其中 \(c\) 是条件，例如文本提示、类别标签、草图或其他控制信息。</p>
          <p>判别模型像是在问：“这个样本属于哪一类？”生成模型像是在问：“什么样的样本会自然出现？”前者更关注决策边界，后者更关注数据分布。两者都很重要，但思维方式不同。</p>
          <p>举一个简单例子。假设我们只关心身高和性别。判别模型可能学习 \(p(\text{性别}|\text{身高})\)，即根据身高判断性别；生成模型可能学习 \(p(\text{身高}|\text{性别})\)，即给定性别后生成合理身高，也可能学习联合分布 \(p(\text{身高},\text{性别})\)。</p>
          <p>在现代生成式 AI 里，条件生成非常常见。文本到图像模型学习的是“在给定文本条件下，图片应该服从什么分布”；语言模型学习的是“在给定前文条件下，下一个 token 应该服从什么分布”。这些都是生成建模思想的延伸。</p>
        </section>

        <section class="article-section">
          <h2>5. 似然：用数据反过来评价参数好不好</h2>
          <p>如果我们已经选择了模型族 \(p_\theta(x)\)，接下来要决定参数 \(\theta\)。什么样的参数才算好？一个自然标准是：如果真实观测数据在这个参数下出现的概率很高，那么这个参数就比较合理。</p>
          <p>假设训练集是：</p>
          <div class="equation">\[
            D=\{x^{(1)},x^{(2)},\ldots,x^{(N)}\}.
          \]</div>
          <p>如果每个样本独立来自同一个分布，那么在参数 \(\theta\) 下观测到整个数据集的概率密度是：</p>
          <div class="equation">\[
            p(D;\theta)
            =
            \prod_{i=1}^{N}p_\theta(x^{(i)}).
          \]</div>
          <p>当我们把它看成关于参数 \(\theta\) 的函数时，它叫似然函数：</p>
          <div class="equation">\[
            L(\theta)=p(D;\theta).
          \]</div>
          <p>注意这里有一个视角转换。概率密度 \(p_\theta(x)\) 通常把 \(x\) 当变量，\(\theta\) 固定；似然 \(L(\theta)\) 则把数据 \(D\) 固定，把 \(\theta\) 当变量。我们问的是：哪一组参数最能解释已经发生的这些数据？</p>
          <p>最大似然估计就是选择使似然最大的参数：</p>
          <div class="equation">\[
            \theta^\star=\arg\max_\theta p(D;\theta).
          \]</div>
          <p>实际计算时通常最大化对数似然：</p>
          <div class="equation">\[
            \log p(D;\theta)
            =
            \sum_{i=1}^{N}\log p_\theta(x^{(i)}).
          \]</div>
          <p>为什么取对数？因为乘积会变成求和，计算更稳定，推导也更方便。而且对数函数单调递增，最大化 \(p(D;\theta)\) 和最大化 \(\log p(D;\theta)\) 得到的参数相同。</p>
          <p>从直觉上说，最大似然估计会惩罚那些让真实样本概率很低的参数。如果某个参数把概率质量放在远离训练数据的地方，那么训练集在它下面出现的似然就很低，它自然不是好参数。</p>
        </section>

        <section class="article-section">
          <h2>6. 正态分布例子：最大似然到底估计出了什么？</h2>
          <p>继续看身高例子。我们假设身高服从正态分布：</p>
          <div class="equation">\[
            x\sim\mathcal{N}(\mu,\sigma^2).
          \]</div>
          <p>现在训练集给了我们 \(N\) 个身高值。最大似然估计要找到最能解释这些身高值的 \(\mu\) 和 \(\sigma\)。结果是：</p>
          <div class="equation">\[
            \hat{\mu}=\frac{1}{N}\sum_{i=1}^{N}x^{(i)}.
          \]</div>
          <div class="equation">\[
            \hat{\sigma}^2=
            \frac{1}{N}\sum_{i=1}^{N}
            \left(x^{(i)}-\hat{\mu}\right)^2.
          \]</div>
          <p>也就是说，最大似然给出的正态分布均值就是样本均值，方差就是样本平均平方偏差。这个结论看起来很自然，但它不是凭直觉拍脑袋来的，而是从最大化对数似然推导出来的。</p>
          <p>这说明一件很重要的事：参数估计不是随便调参数，而是有明确目标函数。我们选择参数，是为了让观测数据在模型下尽可能“合理”。</p>
          <p>当我们把 \(\hat{\mu}\) 和 \(\hat{\sigma}\) 放回正态分布后，就得到了一个完整的生成模型。它可以做两件事：</p>
          <ol>
            <li>生成新数据：从 \(\mathcal{N}(\hat{\mu},\hat{\sigma}^2)\) 采样。</li>
            <li>计算概率：通过密度函数或累积分布函数计算某些事件的概率。</li>
          </ol>
          <p>这正是书中步骤 2.4 强调的两类用途。生成模型不是只能“生成”，它还可以帮助我们进行概率判断。比如估计身高大于某个阈值的概率，本质上也是利用学到的分布。</p>
        </section>

        <section class="article-section">
          <h2>6.5 一个完整的生成建模流程</h2>
          <p>现在把前面的内容压成一个完整流程。以后遇到任何生成模型，都可以先按这几步拆解。</p>
          <p><strong>第一步，确定数据对象。</strong>我们要建模的随机变量是什么？如果是身高，\(x\) 是一个实数；如果是图片，\(x\) 是一个高维张量；如果是文本，\(x\) 是 token 序列。不同数据对象决定了模型需要表达的结构。</p>
          <p><strong>第二步，承认真实分布未知。</strong>我们假设训练样本来自某个 \(p_{\mathrm{data}}(x)\)，但我们不知道它的解析式。这个假设不是为了把问题复杂化，而是为了提醒自己：训练集只是样本，不是总体。</p>
          <p><strong>第三步，选择模型族。</strong>如果数据简单，可以选择正态分布；如果数据多峰，可以选择 GMM；如果数据是图像，就需要神经网络构造复杂分布。这个选择决定了模型的表达能力，也决定了训练和采样方式。</p>
          <p><strong>第四步，定义训练目标。</strong>最经典的是最大似然：</p>
          <div class="equation">\[
            \max_\theta
            \sum_{i=1}^{N}
            \log p_\theta(x^{(i)}).
          \]</div>
          <p>如果精确似然不好算，就可能使用变分下界、对抗目标、score matching、denoising objective 等替代目标。替代目标看起来不同，但通常仍然服务于同一个目的：让模型分布接近真实数据分布。</p>
          <p><strong>第五步，优化参数。</strong>对于一维正态分布，最大似然有解析解；对于神经网络生成模型，通常用梯度下降。优化算法本身不是生成模型的本质，它只是找到好参数的工具。</p>
          <p><strong>第六步，使用模型。</strong>训练完成后，我们可以从模型中采样，也可以用模型做概率评价、异常检测、缺失数据补全或条件生成。模型的价值来自它学到的分布，而不仅仅是生成一张好看的图。</p>
          <p>这六步看起来朴素，但非常通用。后续学 Diffusion 时，也可以这样对应：数据对象是图片或 latent；真实分布是图片数据分布；模型族是由 U-Net/Transformer 参数化的反向过程或 score；训练目标是噪声预测或变分目标；优化用梯度下降；使用时从噪声采样并逐步生成。</p>
        </section>

        <section class="article-section">
          <h2>7. 拟合分布和最大似然为什么是一回事？</h2>
          <p>上面我们说最大似然会选出能解释数据的参数。但它和“拟合真实分布”之间是什么关系？这里需要引入 KL 散度。</p>
          <p>KL 散度可以衡量两个分布之间的差异。对于真实分布 \(p_{\mathrm{data}}(x)\) 和模型分布 \(p_\theta(x)\)，一个常见方向是：</p>
          <div class="equation">\[
            D_{\mathrm{KL}}
            \left(
            p_{\mathrm{data}}
            \,\|\,p_\theta
            \right)
            =
            \int
            p_{\mathrm{data}}(x)
            \log
            \frac{p_{\mathrm{data}}(x)}{p_\theta(x)}
            dx.
          \]</div>
          <p>KL 散度越小，说明模型分布越接近真实分布。当两个分布完全相同时，KL 散度为 0。</p>

          <p>如果随机变量是离散的，KL 散度写成求和形式：</p>
          <div class="equation">\[
            D_{\mathrm{KL}}(p\|q)
            =
            \sum_x p(x)\log\frac{p(x)}{q(x)}.
          \]</div>
          <p>这个式子可以逐项理解。每个 \(x\) 都由 \(p(x)\) 加权，这表示我们主要关心真实分布 \(p\) 经常出现的位置；后面的 \(\log\frac{p(x)}{q(x)}\) 则比较真实概率和模型概率。如果某个位置真实概率很高，但模型概率很低，那么这一项会变大，KL 散度会惩罚模型。</p>

          <p>用书中的硬币例子可以这样理解。真实硬币正面概率是 0.7，反面概率是 0.3。如果模型 \(q_1\) 认为正反面都是 0.5，它没有完全正确，但还算接近，所以 KL 较小。如果模型 \(q_2\) 认为正面只有 0.2、反面有 0.8，它和真实分布方向几乎相反，所以 KL 较大。如果模型 \(q_3\) 完全等于真实分布，那么 KL 为 0。</p>
          <p>KL 散度有一个容易误解的地方：它不是普通距离。普通距离通常满足对称性，比如从 A 到 B 的距离等于从 B 到 A 的距离；但 KL 散度一般不对称：</p>
          <div class="equation">\[
            D_{\mathrm{KL}}(p\|q)
            \neq
            D_{\mathrm{KL}}(q\|p).
          \]</div>
          <p>为什么会不对称？因为 \(D_{\mathrm{KL}}(p\|q)\) 是用 \(p(x)\) 加权的，它更关心真实分布 \(p\) 常出现的位置。如果在某个真实高概率区域，模型 \(q\) 给了极低概率，惩罚会非常重。反过来，\(D_{\mathrm{KL}}(q\|p)\) 又会用 \(q(x)\) 加权，关注点就变了。</p>
          <p>一个极端情况最能说明问题：如果某个位置 \(p(x)>0\)，但 \(q(x)=0\)，那么 \(\log\frac{p(x)}{q(x)}\) 会发散，KL 变成无穷大。直觉上，这是因为真实世界明明会出现某类样本，但模型说它绝不可能出现，这个错误非常严重。</p>

          <p>我们想让 \(p_\theta(x)\) 接近 \(p_{\mathrm{data}}(x)\)，也就是最小化：</p>
          <div class="equation">\[
            \arg\min_\theta
            D_{\mathrm{KL}}
            \left(
            p_{\mathrm{data}}
            \,\|\,p_\theta
            \right).
          \]</div>
          <p>展开 KL 散度：</p>
          <div class="equation">\[
            D_{\mathrm{KL}}
            =
            \int p_{\mathrm{data}}(x)\log p_{\mathrm{data}}(x)\,dx
            -
            \int p_{\mathrm{data}}(x)\log p_\theta(x)\,dx.
          \]</div>
          <p>第一项只和真实分布有关，不包含 \(\theta\)，所以优化参数时可以看成常数。真正影响 \(\theta\) 的是第二项：</p>
          <div class="equation">\[
            \arg\min_\theta D_{\mathrm{KL}}
            =
            \arg\max_\theta
            \int p_{\mathrm{data}}(x)\log p_\theta(x)\,dx.
          \]</div>
          <p>但我们不知道 \(p_{\mathrm{data}}(x)\) 的完整形式，只能用训练样本近似这个期望。于是得到：</p>
          <div class="equation">\[
            \int p_{\mathrm{data}}(x)\log p_\theta(x)\,dx
            \approx
            \frac{1}{N}\sum_{i=1}^{N}
            \log p_\theta(x^{(i)}).
          \]</div>
          <p>右边就是平均对数似然。于是我们得到一个非常重要的结论：</p>
          <div class="insight-box">在样本足够多时，最大化训练数据的对数似然，可以理解为最小化真实数据分布和模型分布之间的 KL 散度。</div>
          <p>这就是“为什么最大似然是在拟合分布”的理论连接。书中步骤 5.1.3 也强调了这一点：最大似然估计可以从最小化 KL 散度的角度得到。</p>
          <p>所以，当我们训练生成模型时，最大似然不是单纯让训练样本概率变大，它背后的目标是让模型分布更接近真实分布。训练集只是我们接触真实分布的方式。</p>
        </section>

        <section class="article-section">
          <h2>8. 生成模型的两种能力：采样和评价</h2>
          <p>一个明确的概率生成模型通常有两种能力。</p>
          <p><strong>第一，采样。</strong>如果我们能从 \(p_\theta(x)\) 中抽样，就能得到新数据：</p>
          <div class="equation">\[
            x_{\mathrm{new}}\sim p_\theta(x).
          \]</div>
          <p>身高模型中，采样就是从一个正态分布随机生成新身高。图像模型中，采样就是生成新图片。Diffusion 中，从高斯噪声开始逐步去噪，本质上也是一种采样过程。</p>

          <p>书中用真实身高数据和生成身高数据的直方图来说明这一点：生成模型不是要求生成样本逐个等于训练样本，而是要求生成样本整体形成的分布和真实数据分布大体重合。换句话说，判断生成模型好不好，不能只看某一个样本，而要看模型采样出来的一批样本是否具有真实数据的统计特征。</p>
          <p><strong>第二，概率评价。</strong>如果我们能计算 \(p_\theta(x)\) 或相关概率，就能判断某个样本在模型下有多合理。例如，身高 172 厘米在模型下密度较高，身高 230 厘米密度极低。对于图像，理想情况下自然图片应该有高概率，随机噪声应该有低概率。</p>
          <p>不同生成模型对这两种能力的支持程度不同。有些模型很容易采样，但不容易计算精确似然，比如 GAN。有些模型可以计算或近似似然，比如 VAE、Flow、Autoregressive Model。有些模型训练和采样路径比较特殊，比如 Diffusion。</p>
          <p>这也解释了为什么生成模型有很多类别。它们都想学习数据分布，但选择了不同的表达方式、训练目标和采样方法。</p>
        </section>

        <section class="article-section">
          <h2>9. 从一维身高到高维图像，困难在哪里？</h2>
          <p>一维身高数据可以用正态分布近似，这是因为它结构简单。一个均值和一个标准差就能描述主要形状。但图像分布远远复杂得多。</p>
          <p>首先，图像是高维对象。一个样本不是一个数字，而是几十万维的向量。高维空间里，大部分点都没有意义。随机采样像素几乎一定得到噪声图。</p>
          <p>其次，图像分布有复杂依赖。相邻像素有关联，局部纹理和整体语义有关联，物体形状和背景也有关联。简单正态分布无法表达这些关系。</p>
          <p>再次，图像分布通常是多峰的。猫、狗、车、人脸、风景都属于自然图片，但它们集中在高维空间中的不同区域。一个单峰高斯不可能同时很好地描述这些复杂类别。</p>
          <p>因此，现代生成模型需要更强的模型族。神经网络的作用，就是提供足够强的函数表达能力。它可以把简单噪声映射成复杂样本，也可以预测去噪方向，还可以参数化条件分布。</p>
          <p>但是不要被复杂网络吓住。无论模型多复杂，底层问题仍然是：</p>
          <div class="equation">\[
            \text{Find } p_\theta(x)
            \text{ such that }
            p_\theta(x)\approx p_{\mathrm{data}}(x).
          \]</div>
          <p>这就是生成模型的共同主线。</p>
        </section>

        <section class="article-section">
          <h2>10. 几类生成模型在同一张地图上的位置</h2>
          <p>现在我们可以把常见生成模型放到同一张概念地图里。它们不是互不相干的名词，而是围绕同一个分布建模目标给出的不同方案。</p>
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>模型类型</th><th>核心想法</th><th>和分布拟合的关系</th></tr>
              </thead>
              <tbody>
                <tr><td>正态分布模型</td><td>用均值和方差描述一维或多维数据</td><td>最简单的显式概率模型，可以直接最大似然估计。</td></tr>
                <tr><td>高斯混合模型</td><td>用多个高斯分布叠加描述多峰数据</td><td>比单个高斯更灵活，引入潜变量表示样本来自哪个成分。</td></tr>
                <tr><td>VAE</td><td>用潜变量和神经网络生成数据</td><td>通过 ELBO 近似最大化似然，学习潜变量分布和解码器。</td></tr>
                <tr><td>GAN</td><td>生成器和判别器对抗训练</td><td>不直接写出似然，而是通过对抗方式让生成分布接近真实分布。</td></tr>
                <tr><td>Flow</td><td>用可逆变换把简单分布变成复杂分布</td><td>可以精确计算密度，直接进行最大似然训练。</td></tr>
                <tr><td>Diffusion</td><td>先把数据逐步加噪，再学习反向去噪</td><td>通过噪声条件分布、变分目标或 score 学习数据分布。</td></tr>
              </tbody>
            </table>
          </div>
          <p>如果只记模型名字，很容易迷路。更好的学习方式是每遇到一个模型，就问三个问题：</p>
          <ol>
            <li>它如何表达 \(p_\theta(x)\)？</li>
            <li>它用什么目标让 \(p_\theta(x)\) 接近 \(p_{\mathrm{data}}(x)\)？</li>
            <li>训练完成后，它如何从 \(p_\theta(x)\) 中采样？</li>
          </ol>
          <p>这三个问题能把大多数生成模型串起来。</p>
        </section>

        <section class="article-section">
          <h2>11. 常见误解</h2>
          <h3>误解一：生成模型就是会画图的模型</h3>
          <p>画图只是生成模型的一种应用。生成模型可以生成图像、音频、文本、分子结构，也可以生成一维身高数据。只要目标是学习数据分布并从中产生新样本，就属于生成建模的范畴。</p>

          <h3>误解二：生成模型只需要记住训练集</h3>
          <p>如果模型只是记住训练样本，它无法真正生成新数据。好的生成模型要学习总体规律，而不是复制样本。它生成的新样本应该没有出现在训练集中，但仍符合训练数据背后的分布。</p>

          <h3>误解三：概率密度越大就等于概率越大</h3>
          <p>连续变量中，密度不是单点概率。真正的概率来自区间上的积分。这个区别在理解似然和概率密度时非常重要。</p>

          <h3>误解四：最大似然只是一个工程损失</h3>
          <p>最大似然背后有明确的分布拟合意义。它可以看成用训练样本近似真实分布下的期望，从而最小化真实分布到模型分布的 KL 散度。</p>

          <h3>误解五：深度生成模型和简单概率模型完全无关</h3>
          <p>深度生成模型只是把 \(p_\theta(x)\) 做得更复杂、更有表达力。它们仍然继承了概率建模的基本问题：分布是什么，参数是什么，训练目标是什么，采样过程是什么。</p>
        </section>

        <section class="article-section">
          <h2>12. 读完本文后应该掌握什么？</h2>
          <p>如果这篇文章读懂了，你应该能用自己的话回答下面几个问题。</p>
          <ol>
            <li>为什么训练集可以看成从真实数据分布 \(p_{\mathrm{data}}(x)\) 中采样得到？</li>
            <li>为什么生成模型不是直接背数据，而是学习一个模型分布 \(p_\theta(x)\)？</li>
            <li>为什么 \(p_\theta(x)\approx p_{\mathrm{data}}(x)\) 后，从 \(p_\theta(x)\) 采样就能生成像真实数据的新样本？</li>
            <li>似然 \(p(D;\theta)\) 和概率密度 \(p_\theta(x)\) 的视角有什么区别？</li>
            <li>为什么最大似然估计可以看成分布拟合？</li>
            <li>为什么从身高这种一维生成模型到图像生成模型，困难主要来自高维和复杂依赖？</li>
          </ol>
          <p>这些问题就是后续学习 Diffusion 的地基。Diffusion 的前向加噪、反向去噪、score、ELBO、Flow Matching 等概念，本质上都建立在“学习数据分布”这件事上。</p>
        </section>

        <section class="article-section">
          <h2>13. 多轮自检：初学者最容易卡在哪里？</h2>
          <p>写到这里，我会用一个完全初学者的视角反复检查：有没有某句话听起来懂了，但其实还不知道为什么？有没有某个公式可以背下来，但不知道它解决了什么问题？下面不是为了凑问答，而是给读者一个回看索引。每个问题后面都标出应该回到哪里找答案。</p>

          <h3>第一轮：定义有没有真正落地？</h3>
          <p><strong>问题一：为什么不能直接说“生成模型就是学习训练集”？</strong>因为训练集只是有限样本。一个模型如果只记住训练集，它最多能复读旧样本，不能理解总体规律。生成模型真正想学的是训练集背后的数据分布。对于身高数据，训练集中可能没有 173.9 厘米这个值，但如果总体分布支持这个身高，模型就应该能生成它。对于图片也是一样，模型生成的新图片不应该只是训练图像的复制品，而应该是符合真实图片分布的新样本。卡住时回到第 1 节和第 4.5 节。</p>
          <p><strong>问题二：为什么一定要引入概率分布？不能只训练一个函数吗？</strong>如果只训练一个确定性函数，输入相同就输出相同，很难描述“可能性”。生成问题天然带有不确定性：同一句提示词可以对应许多合理图片，同一个类别可以对应许多不同样本。概率分布提供了一种语言，告诉我们哪些样本更可能出现，哪些样本不太可能出现。没有分布，就很难严格讨论“像真实数据”这件事。卡住时回到第 2 节和第 4 节。</p>
          <p><strong>问题三：什么叫“从模型里采样”？</strong>采样不是让模型复制某个训练样本，而是按照模型分布的概率规律随机产生一个新样本。身高模型里，采样就是从正态分布抽一个身高；图像生成里，采样就是从学到的高维图像分布里抽一个图像。卡住时回到第 8 节。</p>

          <h3>第二轮：训练目标有没有讲通？</h3>
          <p><strong>问题四：似然里的概率密度很小，为什么还要最大化它？</strong>在连续高维空间里，单个样本的密度值可能非常小，这是正常的。最大似然不是关心绝对数值好不好看，而是比较不同参数下训练数据的相对合理性。参数 A 下训练数据密度更高，说明 A 比参数 B 更能解释这些观测。实际计算时我们用对数似然，把很多小数的乘积变成求和，也避免数值下溢。卡住时回到第 5 节和第 6 节。</p>
          <p><strong>问题五：KL 散度到底在惩罚什么？</strong>对于 \(D_{\mathrm{KL}}(p_{\mathrm{data}}\|p_\theta)\)，它重点惩罚模型在真实数据高概率区域给出低概率。也就是说，真实世界常见的东西，模型不能说它罕见；真实世界会出现的模式，模型不能完全漏掉。这和生成模型的目标一致：模型必须覆盖真实数据的重要区域。卡住时回到第 7 节，尤其是硬币例子和 KL 展开。</p>
          <p><strong>问题六：最大似然和 KL 散度是不是两个不同目标？</strong>在这里它们是同一件事的两种看法。最大似然从样本角度出发：让训练数据在模型下尽可能合理。KL 散度从分布角度出发：让模型分布接近真实分布。由于真实分布未知，我们用训练样本近似真实分布下的期望，于是最小化 \(D_{\mathrm{KL}}(p_{\mathrm{data}}\|p_\theta)\) 就变成最大化平均对数似然。卡住时回到第 7 节最后的推导。</p>

          <h3>第三轮：能不能迁移到现代生成模型？</h3>
          <p><strong>问题七：为什么一个简单正态分布也算生成模型？</strong>因为它满足生成模型的定义：它对数据分布进行建模，并且可以从模型分布中采样新数据。不要把“生成模型”只理解成大型神经网络。正态分布模型、高斯混合模型、VAE、Flow、Diffusion 都在同一条线上，只是表达能力和训练方式不同。卡住时回到第 3 节、第 6 节和第 10 节。</p>
          <p><strong>问题八：如果模型分布接近真实分布，为什么单个生成样本仍可能不好？</strong>分布相近说的是整体统计规律，而不是每一次采样都完美。一个模型可能总体上覆盖了真实数据的主要模式，但某些采样点仍然落在低质量区域。评价生成模型时，不能只看一张图，也不能只看一个数字，而要同时看样本质量、多样性、分布覆盖和是否过拟合。卡住时回到第 8 节和第 9 节。</p>
          <p><strong>问题九：这篇文章和下一篇 Diffusion 有什么关系？</strong>Diffusion 也是生成模型，所以它仍然要学习 \(p_{\mathrm{data}}(x)\)。区别在于它不直接写出一个简单的 \(p_\theta(x)\)，而是构造一条从数据到噪声、再从噪声回到数据的路径。理解了本文的分布、似然、KL、采样之后，再看 Diffusion 的前向加噪、反向去噪、ELBO 和 score，就会知道它们不是孤立技巧，而是在服务同一个目标：让模型分布接近真实数据分布。卡住时回到第 10 节，然后进入下一篇。</p>
        </section>

        <section class="article-section references">
          <h2>参考资料</h2>
          <ul>
            <li>斋藤康毅 著，郑明智 译，《深度学习入门5：生成模型》，人民邮电出版社，2025。本文主要参考步骤 1“正态分布”、步骤 2“最大似然估计”和步骤 5.1“KL 散度”；文中的总体/样本、参数估计、真实与生成分布、KL 硬币例子均为根据书中相关图意改写成文字说明。</li>
          </ul>
        </section>
      `,
    },
    {
      slug: "diffusion-model-foundations",
      title: "Diffusion Model 基础理论：从生成建模到 DDPM、Score 与 Flow Matching",
      date: "2026-05-21",
      category: "学习笔记",
      summary:
        "一篇面向零生成模型理论基础读者的 Diffusion Model 长文。我们从概率建模和高斯分布讲起，完整推导前向加噪、后验均值、DDPM 训练目标、score matching、SDE/ODE、DDIM 与 Flow Matching 的关系。",
      tags: [
        "Diffusion Model",
        "Generative Model",
        "Machine Learning",
        "Probability",
        "DDPM",
        "Score Matching",
        "Flow Matching",
        "SDE",
        "ODE"
      ],
      cover: "cover-b",
      body: String.raw`
        <p class="lead">这篇文章的目标是：假设读者没有系统学过生成模型，也不默认你已经熟悉 VAE、GAN、score matching 或随机微分方程。我们从“生成模型到底想学什么”开始，逐步建立 Diffusion Model 的数学对象、训练目标和采样方法。读完后，你应该能带着清楚的问题去读 DDPM、DDIM、score-based SDE、flow matching 这些论文，而不是只记住几个公式。</p>

        <section class="article-section">
          <h2>0. 阅读路线：为什么一篇文章要讲这么长？</h2>
          <p>Diffusion Model 容易被讲得很玄：从噪声生成图片、U-Net 预测噪声、score 是梯度、SDE 可以反向走、Flow Matching 和 Diffusion 等价。每一句都对，但如果缺少中间层，初学者会觉得这些概念像从天上掉下来。</p>
          <p>所以本文采用一条比较慢、但更稳的路线：</p>
          <ol>
            <li>先讲生成模型的目标：我们到底要拟合什么分布。</li>
            <li>再讲为什么高斯噪声是一个好的起点：因为它简单、可采样、可计算。</li>
            <li>然后推导 DDPM 的前向过程：为什么 \(q(x_t|x_0)\) 有闭式形式。</li>
            <li>再推导反向后验：为什么 \(q(x_{t-1}|x_t,x_0)\) 是高斯，均值和方差从哪里来。</li>
            <li>接着解释训练目标：为什么预测噪声 \(\epsilon\) 会变成一个简单的 MSE。</li>
            <li>最后把 DDPM、score matching、SDE/ODE、DDIM、flow matching 连起来。</li>
          </ol>
          <p>本文会尽量把每一步的“为什么”说清楚。公式不是为了显得高级，而是为了把直觉固定下来：你知道每个符号是什么意思，才知道论文里的省略步骤在省略什么。</p>

          <figure class="source-figure">
            <img src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/generative-overview.png" alt="生成模型类型对比图" loading="lazy" />
            <figcaption>参考图：GAN、VAE、Flow-based model 和 Diffusion model 的结构对比。Diffusion 可以看成从数据到噪声、再从噪声回到数据的建模路线。图片来源：Lilian Weng, What are Diffusion Models?</figcaption>
          </figure>

          <h3>本文符号表</h3>
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>符号</th><th>含义</th><th>第一次见到时应该怎么想</th></tr>
              </thead>
              <tbody>
                <tr><td>\(x_0\)</td><td>真实数据样本</td><td>一张真实图片，或一个真实数据向量。</td></tr>
                <tr><td>\(x_t\)</td><td>第 \(t\) 个噪声等级下的样本</td><td>它不是新图片，而是由 \(x_0\) 加噪得到的中间状态。</td></tr>
                <tr><td>\(\epsilon\)</td><td>标准高斯噪声</td><td>训练时我们亲手采样，所以知道答案；模型要学会把它预测出来。</td></tr>
                <tr><td>\(\beta_t\)</td><td>第 \(t\) 步加噪强度</td><td>每一步往图像里塞多少新噪声。</td></tr>
                <tr><td>\(\alpha_t=1-\beta_t\)</td><td>第 \(t\) 步保留的信号比例</td><td>为了让尺度稳定，保留信号和加入噪声要一起设计。</td></tr>
                <tr><td>\(\bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s\)</td><td>从第 0 步到第 \(t\) 步累计保留的信号比例</td><td>它控制 \(x_t\) 里还剩多少 \(x_0\) 的影子。</td></tr>
                <tr><td>\(q\)</td><td>固定的前向加噪分布</td><td>不是神经网络学出来的，而是我们人为规定的。</td></tr>
                <tr><td>\(p_\theta\)</td><td>模型学习的反向生成分布</td><td>真正要训练的东西，用参数 \(\theta\) 表示。</td></tr>
                <tr><td>\(\nabla_x\log p_t(x)\)</td><td>score</td><td>当前噪声等级下，往哪里移动会更像数据。</td></tr>
                <tr><td>\(v_\theta(x,t)\)</td><td>Flow Matching 的速度场</td><td>把样本看成粒子，速度场告诉粒子如何从噪声流向数据。</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        <section class="article-section">
          <h2>1. 生成模型到底在学什么？</h2>
          <p>我们先从一个最普通的问题开始：什么叫“生成一张图片”？在数学上，一张图片可以看成一个很长的向量。假设图片大小是 \(H \times W\)，有 3 个颜色通道，那么它可以表示为：</p>
          <div class="equation">\[
            x \in \mathbb{R}^{H \times W \times 3}.
          \]</div>
          <p>如果我们把它拉平成一维向量，也可以写成 \(x \in \mathbb{R}^d\)。真实世界中所有“合理图片”并不是均匀分布在整个 \(\mathbb{R}^d\) 中。随机采样一个像素向量，大概率只是花屏噪声。真正像图片的点，只占据高维空间中非常小、非常有结构的一部分。</p>
          <p>我们把真实数据背后的未知分布记作：</p>
          <div class="equation">\[
            x \sim p_{\mathrm{data}}(x).
          \]</div>
          <p>训练集就是从这个分布里抽出来的一些样本。生成模型的目标，是构造一个参数化分布 \(p_\theta(x)\)，让它尽量接近 \(p_{\mathrm{data}}(x)\)。如果模型足够好，我们从 \(p_\theta(x)\) 采样，就能得到看起来像真实数据的新样本。</p>
          <p>这里有两个困难：</p>
          <ol>
            <li><strong>真实数据分布复杂。</strong>图片分布包含边缘、纹理、物体结构、语义关系，远不是一个简单高斯能描述的。</li>
            <li><strong>高维概率密度难以直接写出。</strong>我们很难直接给每张图片一个可归一化的概率密度，并且还能高效采样。</li>
          </ol>
          <p>不同生成模型在这里走了不同路线。GAN 避开显式概率密度，训练一个生成器和判别器对抗；VAE 引入潜变量并最大化变分下界；Flow 模型构造可逆变换，让密度可以用变量替换公式精确计算。Diffusion Model 的思路是：不直接从复杂分布采样，而是先把复杂分布慢慢破坏成简单高斯，再学习反向修复。</p>
        </section>

        <section class="article-section">
          <h2>2. 为什么噪声分布适合作为生成起点？</h2>
          <p>标准高斯分布 \( \mathcal{N}(0,I) \) 有三个很重要的优点。</p>
          <p>第一，它非常容易采样。我们可以用随机数生成器直接得到 \(\epsilon \sim \mathcal{N}(0,I)\)。</p>
          <p>第二，它的概率密度形式简单：</p>
          <div class="equation">\[
            p(\epsilon)=\frac{1}{(2\pi)^{d/2}}\exp\left(-\frac{1}{2}\|\epsilon\|^2\right).
          \]</div>
          <p>第三，高斯分布在线性变换下仍然是高斯。这个性质会让 Diffusion 的前向过程可以被精确推导。比如如果 \(z\sim \mathcal{N}(0,I)\)，那么 \(az+b\) 也是高斯，均值是 \(b\)，方差是 \(a^2I\)。</p>
          <p>Diffusion 的基本想法可以用一句话概括：</p>
          <div class="insight-box">先设计一条容易计算的前向路径，把真实数据分布逐渐变成高斯噪声；然后训练模型学习这条路径的反方向。</div>
          <p>注意这里的“前向路径”不是自然发生的物理过程，而是我们人为设计的训练工具。它的作用是给生成问题搭一座桥：桥的一端是真实数据，另一端是容易采样的高斯噪声。</p>
        </section>

        <section class="article-section">
          <h2>2.5 高斯工具箱：后面所有推导都靠它</h2>
          <p>很多 Diffusion 推导看起来复杂，其实反复使用三条高斯事实。先把它们讲清楚，后面公式就不会像魔法。</p>

          <h3>事实一：高斯的线性变换仍然是高斯</h3>
          <p>如果 \(z\sim\mathcal{N}(0,I)\)，令 \(x=az+b\)，其中 \(a\) 是标量，\(b\) 是向量，那么：</p>
          <div class="equation">\[
            x\sim \mathcal{N}(b,a^2I).
          \]</div>
          <p>为什么均值是 \(b\)？因为 \(\mathbb{E}[x]=a\mathbb{E}[z]+b=b\)。为什么方差是 \(a^2I\)？因为方差对缩放的响应是平方：</p>
          <div class="equation">\[
            \mathrm{Var}(az+b)=a^2\mathrm{Var}(z)=a^2I.
          \]</div>
          <p>这条事实解释了为什么 DDPM 可以把“从高斯采样”写成“均值 + 标准差乘噪声”。</p>

          <h3>事实二：独立高斯相加，方差相加</h3>
          <p>如果 \(\epsilon_1,\epsilon_2\sim\mathcal{N}(0,I)\) 且互相独立，那么：</p>
          <div class="equation">\[
            a\epsilon_1+b\epsilon_2\sim\mathcal{N}(0,(a^2+b^2)I).
          \]</div>
          <p>这里的“独立”很关键。独立意味着没有协方差交叉项，因此总方差就是每部分方差的和。DDPM 的闭式加噪公式正是靠这个性质，把很多步噪声合并成一份等价噪声。</p>

          <h3>事实三：两个高斯密度相乘，仍然得到一个未归一化高斯</h3>
          <p>一维情形最容易看。设：</p>
          <div class="equation">\[
            p_1(x)\propto \exp\left[-\frac{(x-\mu_1)^2}{2\sigma_1^2}\right],
            \qquad
            p_2(x)\propto \exp\left[-\frac{(x-\mu_2)^2}{2\sigma_2^2}\right].
          \]</div>
          <p>相乘后指数相加：</p>
          <div class="equation">\[
            p_1(x)p_2(x)
            \propto
            \exp\left[
            -\frac{1}{2}
            \left(
            \frac{(x-\mu_1)^2}{\sigma_1^2}
            +
            \frac{(x-\mu_2)^2}{\sigma_2^2}
            \right)
            \right].
          \]</div>
          <p>把括号展开，只看 \(x\) 的二次项和一次项：</p>
          <div class="equation">\[
            \frac{(x-\mu_1)^2}{\sigma_1^2}
            +
            \frac{(x-\mu_2)^2}{\sigma_2^2}
            =
            \left(\frac{1}{\sigma_1^2}+\frac{1}{\sigma_2^2}\right)x^2
            -
            2\left(\frac{\mu_1}{\sigma_1^2}+\frac{\mu_2}{\sigma_2^2}\right)x
            +
            \mathrm{const}.
          \]</div>
          <p>令精度，也就是方差的倒数，为 \(\lambda_i=1/\sigma_i^2\)。新高斯的精度是精度相加：</p>
          <div class="equation">\[
            \lambda=\lambda_1+\lambda_2,
            \qquad
            \sigma^2=\frac{1}{\lambda_1+\lambda_2}.
          \]</div>
          <p>新均值是按精度加权的平均：</p>
          <div class="equation">\[
            \mu=\frac{\lambda_1\mu_1+\lambda_2\mu_2}{\lambda_1+\lambda_2}.
          \]</div>
          <p>这条事实后面会直接用来推导 \(q(x_{t-1}|x_t,x_0)\)。你可以把它理解成两个信息源在投票：方差小的信息源更可靠，因此权重更大。</p>
        </section>

        <section class="article-section">
          <h2>3. DDPM 的前向加噪过程</h2>
          <p>DDPM 把前向过程定义成一个 Markov 链：</p>
          <div class="equation">\[
            x_0 \rightarrow x_1 \rightarrow x_2 \rightarrow \cdots \rightarrow x_T.
          \]</div>
          <p>其中 \(x_0\) 是真实数据，\(x_T\) 希望接近标准高斯。Markov 性的意思是：给定 \(x_{t-1}\) 后，\(x_t\) 不再依赖更早的状态：</p>
          <div class="equation">\[
            q(x_t|x_{t-1},x_{t-2},\ldots,x_0)=q(x_t|x_{t-1}).
          \]</div>
          <p>DDPM 选择每一步加入一点高斯噪声：</p>
          <div class="equation">\[
            q(x_t|x_{t-1})=\mathcal{N}\left(x_t;\sqrt{\alpha_t}x_{t-1},\beta_t I\right),
            \qquad \alpha_t=1-\beta_t.
          \]</div>
          <p>这等价于下面的采样形式：</p>
          <div class="equation">\[
            x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{\beta_t}\epsilon_t,\qquad \epsilon_t\sim\mathcal{N}(0,I).
          \]</div>
          <p>为什么要乘 \(\sqrt{\alpha_t}\)，而不是简单写成 \(x_t=x_{t-1}+\mathrm{noise}\)？原因是尺度控制。假设 \(x_{t-1}\) 的方差大约是 1，那么 \(\sqrt{\alpha_t}x_{t-1}\) 的方差约为 \(\alpha_t\)，噪声项 \(\sqrt{\beta_t}\epsilon_t\) 的方差约为 \(\beta_t\)。二者相加的方差约为 \(\alpha_t+\beta_t=1\)。这让数据在加噪过程中不会无限膨胀。</p>

        </section>

        <section class="article-section">
          <h2>4. 关键闭式公式：直接从 \(x_0\) 得到 \(x_t\)</h2>
          <p>如果每次训练都从 \(x_0\) 一步一步采样到 \(x_t\)，会很麻烦。幸运的是，DDPM 的前向过程有一个闭式公式：</p>
          <div class="equation">\[
            q(x_t|x_0)=\mathcal{N}\left(x_t;\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I\right),
            \qquad \bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s.
          \]</div>
          <p>等价采样形式是：</p>
          <div class="equation">\[
            x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I).
          \]</div>
          <p>下面完整推导。我们先看两步。</p>
          <div class="equation">\[
            x_1=\sqrt{\alpha_1}x_0+\sqrt{\beta_1}\epsilon_1.
          \]</div>
          <div class="equation">\[
            x_2=\sqrt{\alpha_2}x_1+\sqrt{\beta_2}\epsilon_2.
          \]</div>
          <p>把第一式代入第二式：</p>
          <div class="equation">\[
            x_2=\sqrt{\alpha_2\alpha_1}x_0+\sqrt{\alpha_2\beta_1}\epsilon_1+\sqrt{\beta_2}\epsilon_2.
          \]</div>
          <p>两个独立标准高斯的线性组合仍然是高斯。噪声总方差等于系数平方和：</p>
          <div class="equation">\[
            \alpha_2\beta_1+\beta_2=\alpha_2(1-\alpha_1)+(1-\alpha_2)=1-\alpha_1\alpha_2.
          \]</div>
          <p>于是：</p>
          <div class="equation">\[
            x_2=\sqrt{\alpha_1\alpha_2}x_0+\sqrt{1-\alpha_1\alpha_2}\epsilon.
          \]</div>
          <p>一般 \(t\) 步可以用归纳法。假设：</p>
          <div class="equation">\[
            x_{t-1}=\sqrt{\bar{\alpha}_{t-1}}x_0+\sqrt{1-\bar{\alpha}_{t-1}}\epsilon.
          \]</div>
          <p>代入 \(x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{\beta_t}\epsilon_t\)。信号项变为：</p>
          <div class="equation">\[
            \sqrt{\alpha_t}\sqrt{\bar{\alpha}_{t-1}}x_0=\sqrt{\bar{\alpha}_t}x_0.
          \]</div>
          <p>噪声方差变为：</p>
          <div class="equation">\[
            \alpha_t(1-\bar{\alpha}_{t-1})+\beta_t
            =\alpha_t-\bar{\alpha}_t+1-\alpha_t
            =1-\bar{\alpha}_t.
          \]</div>
          <p>所以闭式公式成立。</p>
          <div class="insight-box">这个公式是 DDPM 训练能高效进行的关键。训练时我们可以随机抽一个 \(t\)，一次性合成 \(x_t\)，不需要真的执行 \(t\) 次加噪。</div>

          <figure class="source-figure">
            <img src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/diffusion-beta.png" alt="线性和余弦噪声日程对比" loading="lazy" />
            <figcaption>参考图：不同噪声 schedule 下 \(\bar{\alpha}_t\) 的衰减方式。它帮助理解不同时间步中原始信号还保留多少。图片来源：Lilian Weng, What are Diffusion Models?</figcaption>
          </figure>

        </section>

        <section class="article-section">
          <h2>4.5 从信噪比理解时间步</h2>
          <p>闭式公式也可以写成：</p>
          <div class="equation">\[
            x_t=\underbrace{\sqrt{\bar{\alpha}_t}x_0}_{\text{信号}}
            +
            \underbrace{\sqrt{1-\bar{\alpha}_t}\epsilon}_{\text{噪声}}.
          \]</div>
          <p>所以第 \(t\) 步不是一个抽象编号，而是一个噪声等级。衡量噪声等级最自然的量是信噪比：</p>
          <div class="equation">\[
            \mathrm{SNR}(t)=
            \frac{\text{信号方差}}{\text{噪声方差}}
            =
            \frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}.
          \]</div>
          <p>当 \(t\) 很小，\(\bar{\alpha}_t\) 接近 1，SNR 很大，说明样本里主要是原图信息；当 \(t\) 很大，\(\bar{\alpha}_t\) 接近 0，SNR 很小，说明样本几乎全是噪声。很多论文讨论 schedule，本质上就是讨论模型应该在不同 SNR 区间看到多少训练样本，以及采样时如何分配步长。</p>
          <p>线性 \(\beta_t\) schedule 很直观，但不一定让每个噪声等级都被合理利用。cosine schedule 的动机之一，是让 \(\bar{\alpha}_t\) 的衰减更平滑，避免前期或后期变化过快。读 improved DDPM、EDM、flow matching 或 consistency model 时，经常会看到作者重新设计噪声参数化；这些设计通常都可以回到 SNR 来理解。</p>
          <div class="insight-box">看到新的噪声 schedule 时，不要先背公式。先问三个问题：低噪声区有多细？高噪声区有多细？中间 SNR 区间是否给了模型足够训练信号？</div>
        </section>

        <section class="article-section">
          <h2>5. 反向过程：我们真正要学习的东西</h2>
          <p>生成时我们没有 \(x_0\)。我们只有一个随机噪声 \(x_T\sim\mathcal{N}(0,I)\)，希望一步步得到 \(x_{T-1},x_{T-2},\ldots,x_0\)。因此模型定义为：</p>
          <div class="equation">\[
            p_\theta(x_{0:T})=p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}|x_t).
          \]</div>
          <p>真正困难的是 \(q(x_{t-1}|x_t)\)。只看一个带噪样本 \(x_t\)，并不知道它来自哪个干净样本；许多不同的 \(x_0\) 都可能加噪到相似的 \(x_t\)。</p>
          <p>训练时我们知道 \(x_0\)，所以先研究更容易的后验：</p>
          <div class="equation">\[
            q(x_{t-1}|x_t,x_0).
          \]</div>
          <p>这个分布可以精确推导成高斯。然后我们让神经网络学习一个高斯反向转移来近似它。</p>

          <figure class="source-figure">
            <img src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DDPM.png" alt="DDPM 前向扩散和反向去噪示意图" loading="lazy" />
            <figcaption>参考图：DDPM 的前向扩散链和反向去噪链。读图时重点看方向：前向 \(q\) 是人为设计的加噪，反向 \(p_\theta\) 才是模型学习的生成过程。图片来源：Lilian Weng, What are Diffusion Models?</figcaption>
          </figure>

        </section>

        <section class="article-section">
          <h2>6. 后验 \(q(x_{t-1}|x_t,x_0)\) 的推导</h2>
          <p>根据贝叶斯公式：</p>
          <div class="equation">\[
            q(x_{t-1}|x_t,x_0)=\frac{q(x_t|x_{t-1},x_0)q(x_{t-1}|x_0)}{q(x_t|x_0)}.
          \]</div>
          <p>由于前向过程是 Markov 的，\(q(x_t|x_{t-1},x_0)=q(x_t|x_{t-1})\)。分母 \(q(x_t|x_0)\) 与未知变量 \(x_{t-1}\) 无关，所以如果只关心关于 \(x_{t-1}\) 的分布形状，可以写成：</p>
          <div class="equation">\[
            q(x_{t-1}|x_t,x_0)\propto q(x_t|x_{t-1})q(x_{t-1}|x_0).
          \]</div>
          <p>两个因子分别是：</p>
          <div class="equation">\[
            q(x_t|x_{t-1})=\mathcal{N}(x_t;\sqrt{\alpha_t}x_{t-1},\beta_t I),
          \]</div>
          <div class="equation">\[
            q(x_{t-1}|x_0)=\mathcal{N}(x_{t-1};\sqrt{\bar{\alpha}_{t-1}}x_0,(1-\bar{\alpha}_{t-1})I).
          \]</div>
          <p>令 \(x=x_{t-1}\)。高斯密度的指数部分只看和 \(x\) 有关的项：</p>
          <div class="equation">\[
            -\frac{1}{2\beta_t}\|x_t-\sqrt{\alpha_t}x\|^2
            -\frac{1}{2(1-\bar{\alpha}_{t-1})}\|x-\sqrt{\bar{\alpha}_{t-1}}x_0\|^2.
          \]</div>
          <p>展开第一项：</p>
          <div class="equation">\[
            \|x_t-\sqrt{\alpha_t}x\|^2
            =x_t^\top x_t-2\sqrt{\alpha_t}x_t^\top x+\alpha_t x^\top x.
          \]</div>
          <p>展开第二项：</p>
          <div class="equation">\[
            \|x-\sqrt{\bar{\alpha}_{t-1}}x_0\|^2
            =x^\top x-2\sqrt{\bar{\alpha}_{t-1}}x_0^\top x+\bar{\alpha}_{t-1}x_0^\top x_0.
          \]</div>
          <p>去掉和 \(x\) 无关的常数，得到：</p>
          <div class="equation">\[
            -\frac{1}{2}\left[
            \left(\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar{\alpha}_{t-1}}\right)x^\top x
            -2\left(\frac{\sqrt{\alpha_t}}{\beta_t}x_t+\frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}x_0\right)^\top x
            \right].
          \]</div>
          <p>这就是一个高斯分布的标准二次形式。为了看清楚，我们把它写成：</p>
          <div class="equation">\[
            -\frac{1}{2}\left[A x^\top x-2b^\top x\right],
          \]</div>
          <p>其中：</p>
          <div class="equation">\[
            A=
            \frac{\alpha_t}{\beta_t}
            +
            \frac{1}{1-\bar{\alpha}_{t-1}},
            \qquad
            b=
            \frac{\sqrt{\alpha_t}}{\beta_t}x_t
            +
            \frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}x_0.
          \]</div>
          <p>对一个各向同性高斯来说，二次项前的 \(A\) 是精度，也就是方差的倒数。因此后验方差应该是 \(A^{-1}\)。现在把 \(A\) 化简：</p>
          <div class="equation">\[
            A
            =
            \frac{\alpha_t(1-\bar{\alpha}_{t-1})+\beta_t}
            {\beta_t(1-\bar{\alpha}_{t-1})}.
          \]</div>
          <p>分子继续化简：</p>
          <div class="equation">\[
            \alpha_t(1-\bar{\alpha}_{t-1})+\beta_t
            =
            \alpha_t-\alpha_t\bar{\alpha}_{t-1}+1-\alpha_t
            =
            1-\bar{\alpha}_t.
          \]</div>
          <p>所以：</p>
          <div class="equation">\[
            A=
            \frac{1-\bar{\alpha}_t}
            {\beta_t(1-\bar{\alpha}_{t-1})},
            \qquad
            A^{-1}=
            \frac{\beta_t(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}.
          \]</div>
          <p>这就得到后验方差。后验均值来自配方。因为：</p>
          <div class="equation">\[
            A x^\top x-2b^\top x
            =
            A\left\|x-\frac{b}{A}\right\|^2+\mathrm{const},
          \]</div>
          <p>所以均值是 \(A^{-1}b\)。把 \(A^{-1}\) 乘到 \(b\) 上，就会得到两个权重：一个乘 \(x_0\)，一个乘 \(x_t\)。整理后：</p>
          <div class="equation">\[
            q(x_{t-1}|x_t,x_0)=\mathcal{N}\left(x_{t-1};\tilde{\mu}_t(x_t,x_0),\tilde{\beta}_t I\right),
          \]</div>
          <div class="equation">\[
            \tilde{\beta}_t=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t,
          \]</div>
          <div class="equation">\[
            \tilde{\mu}_t(x_t,x_0)
            =
            \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0
            +
            \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t.
          \]</div>
          <p>这个结果值得停下来理解：最优后验均值是 \(x_0\) 和 \(x_t\) 的加权组合。低噪声时 \(x_t\) 已经很接近 \(x_0\)，高噪声时需要更多依赖对干净样本的估计。</p>
        </section>

        <section class="article-section">
          <h2>7. 为什么 DDPM 通常让网络预测噪声？</h2>
          <p>我们已经知道：</p>
          <div class="equation">\[
            x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon.
          \]</div>
          <p>如果网络能预测加入的噪声 \(\epsilon\)，就可以反解出对 \(x_0\) 的估计：</p>
          <div class="equation">\[
            \hat{x}_0=\frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t,t)}{\sqrt{\bar{\alpha}_t}}.
          \]</div>
          <p>再把 \(\hat{x}_0\) 代入 \(\tilde{\mu}_t(x_t,x_0)\)，就得到反向采样的均值。</p>
          <p>训练时，真实噪声 \(\epsilon\) 是我们自己采样出来的，所以可以直接监督：</p>
          <div class="equation">\[
            L_{\mathrm{simple}}(\theta)=
            \mathbb{E}_{x_0,t,\epsilon}
            \left[
            \|\epsilon-\epsilon_\theta(x_t,t)\|^2
            \right].
          \]</div>
          <p>这就是 DDPM 中最常用的简化训练目标。它表面上是一个普通 MSE，但背后对应的是变分下界中每个反向高斯转移的 KL 项。DDPM 论文发现，去掉复杂权重后用这个简单目标训练，图像质量反而很好。</p>
          <div class="insight-box">预测噪声并不是唯一选择。模型也可以预测 \(x_0\)、score、或者 velocity \(v\)。这些参数化之间可以互相转换；不同选择会影响训练稳定性、采样器设计和不同噪声等级下的误差放大方式。</div>
        </section>

        <section class="article-section">
          <h2>7.5 训练算法：把推导翻译成代码会发生什么？</h2>
          <p>理论推导到这里，训练过程其实已经很简单。每一次迭代做四件事：</p>
          <ol>
            <li>从训练集中取一批真实样本 \(x_0\)。</li>
            <li>随机采样时间步 \(t\)，通常从 \(\{1,\ldots,T\}\) 均匀采样。</li>
            <li>采样标准高斯噪声 \(\epsilon\)，用闭式公式合成 \(x_t\)。</li>
            <li>让网络 \(\epsilon_\theta(x_t,t)\) 预测这份噪声，用 MSE 更新参数。</li>
          </ol>
          <p>写成数学形式就是：</p>
          <div class="equation">\[
            t\sim \mathrm{Uniform}\{1,\ldots,T\},
            \qquad
            \epsilon\sim\mathcal{N}(0,I),
          \]</div>
          <div class="equation">\[
            x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,
          \]</div>
          <div class="equation">\[
            \theta \leftarrow
            \theta-\eta\nabla_\theta
            \left\|
            \epsilon-\epsilon_\theta(x_t,t)
            \right\|^2.
          \]</div>
          <p>这个过程看起来像“训练一个去噪网络”，但它比普通图像去噪更强：普通去噪通常固定某个噪声等级，而 Diffusion 会在很多噪声等级上训练同一个网络。因此模型学到的是一族去噪问题，而不是单个去噪任务。</p>

          <h3>\(\epsilon\)-prediction、\(x_0\)-prediction 和 \(v\)-prediction</h3>
          <p>设：</p>
          <div class="equation">\[
            \sigma_t=\sqrt{1-\bar{\alpha}_t},
            \qquad
            a_t=\sqrt{\bar{\alpha}_t}.
          \]</div>
          <p>闭式加噪写成 \(x_t=a_tx_0+\sigma_t\epsilon\)。如果模型预测噪声，就可以得到：</p>
          <div class="equation">\[
            \hat{x}_0=\frac{x_t-\sigma_t\epsilon_\theta(x_t,t)}{a_t}.
          \]</div>
          <p>如果模型直接预测 \(x_0\)，则可以反推出噪声：</p>
          <div class="equation">\[
            \hat{\epsilon}=\frac{x_t-a_t x_{\theta,0}(x_t,t)}{\sigma_t}.
          \]</div>
          <p>很多现代实现还使用 \(v\)-prediction。一个常见定义是：</p>
          <div class="equation">\[
            v=a_t\epsilon-\sigma_t x_0.
          \]</div>
          <p>它把噪声和干净样本混合起来。为什么要这样做？因为在低噪声区，直接预测 \(\epsilon\) 可能让很小的噪声误差被放大；在高噪声区，直接预测 \(x_0\) 又非常困难。\(v\) 参数化在不同噪声等级之间更均衡一些。</p>
          <p>从 \(x_t\) 和 \(v\) 可以恢复 \(x_0\) 与 \(\epsilon\)：</p>
          <div class="equation">\[
            x_0=a_t x_t-\sigma_t v,
            \qquad
            \epsilon=\sigma_t x_t+a_t v.
          \]</div>
          <p>这里不要死背。只要记住 \(x_t=a_tx_0+\sigma_t\epsilon\) 和 \(v=a_t\epsilon-\sigma_tx_0\)，它们组成一个二维旋转式的线性变换，代数上可以互相解出。</p>
        </section>

        <section class="article-section">
          <h2>8. 从噪声预测到 score matching</h2>
          <p>score 定义为概率密度对输入的 log 梯度：</p>
          <div class="equation">\[
            s_t(x)=\nabla_x\log p_t(x).
          \]</div>
          <p>它的直觉是：告诉我们当前位置往哪个方向移动，概率密度上升最快。对高斯分布来说，score 会把点拉回均值附近。</p>
          <p>考虑条件分布：</p>
          <div class="equation">\[
            q(x_t|x_0)=\mathcal{N}(\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I).
          \]</div>
          <p>对 \(x_t\) 求梯度：</p>
          <div class="equation">\[
            \nabla_{x_t}\log q(x_t|x_0)
            =
            -\frac{x_t-\sqrt{\bar{\alpha}_t}x_0}{1-\bar{\alpha}_t}.
          \]</div>
          <p>又因为：</p>
          <div class="equation">\[
            x_t-\sqrt{\bar{\alpha}_t}x_0=\sqrt{1-\bar{\alpha}_t}\epsilon,
          \]</div>
          <p>所以：</p>
          <div class="equation">\[
            \nabla_{x_t}\log q(x_t|x_0)
            =
            -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}.
          \]</div>
          <p>这说明预测噪声 \(\epsilon\) 与预测条件 score 只差一个已知的尺度因子。DDPM 和 score-based model 的核心联系就在这里：二者都在学习 noisy distribution 的方向信息。</p>
          <p>不过论文里真正需要的是边缘分布 \(p_t(x_t)\) 的 score，而不是给定 \(x_0\) 后的条件 score。边缘分布是：</p>
          <div class="equation">\[
            p_t(x_t)=\int q(x_t|x_0)p_{\mathrm{data}}(x_0)dx_0.
          \]</div>
          <p>这个积分表示：所有可能的真实样本 \(x_0\) 都可能加噪得到当前 \(x_t\)。直接求 \(\nabla_{x_t}\log p_t(x_t)\) 很难，因为真实数据分布未知。但 denoising score matching 给了一个可训练的办法：用条件 score 作为监督，学习一个网络 \(s_\theta(x_t,t)\)，使它在期望意义下接近边缘 score。</p>
          <div class="equation">\[
            \mathbb{E}_{p_{\mathrm{data}}(x_0)q(x_t|x_0)}
            \left[
            \left\|
            s_\theta(x_t,t)
            -
            \nabla_{x_t}\log q(x_t|x_0)
            \right\|^2
            \right].
          \]</div>
          <p>直觉是：同一个 \(x_t\) 可能由很多 \(x_0\) 产生，训练时模型看见大量这样的配对后，会学到这些条件方向的平均效果。这个平均方向正是把样本推回高概率区域的边缘 score。</p>
          <div class="insight-box">所以“预测噪声”不是一个随便挑的工程技巧。它是用一个容易构造的监督信号，间接学习每个噪声等级上的 score。</div>

        </section>

        <section class="article-section">
          <h2>9. 变分下界：DDPM 训练目标从哪里来？</h2>
          <p>如果我们希望最大化模型对数据的似然 \(p_\theta(x_0)\)，直接计算通常很困难，因为模型包含整条隐变量链 \(x_{1:T}\)。于是使用变分推断。我们从边缘似然开始：</p>
          <div class="equation">\[
            p_\theta(x_0)=\int p_\theta(x_{0:T})dx_{1:T}.
          \]</div>
          <p>引入前向过程 \(q(x_{1:T}|x_0)\) 作为变分分布：</p>
          <div class="equation">\[
            \log p_\theta(x_0)
            =
            \log \int q(x_{1:T}|x_0)\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}dx_{1:T}.
          \]</div>
          <p>把积分写成期望：</p>
          <div class="equation">\[
            \log p_\theta(x_0)
            =
            \log \mathbb{E}_{q}
            \left[
            \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}
            \right].
          \]</div>
          <p>由于 \(\log\) 是凹函数，Jensen 不等式给出：</p>
          <div class="equation">\[
            \log \mathbb{E}_q[Y]\geq \mathbb{E}_q[\log Y].
          \]</div>
          <p>所以：</p>
          <div class="equation">\[
            \log p_\theta(x_0)
            \geq
            \mathbb{E}_q
            \left[
            \log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}
            \right].
          \]</div>
          <p>这就是 ELBO。DDPM 通常最小化负的 ELBO，因此常写成变分上界 \(L_{\mathrm{vlb}}\)。把前向链和反向链展开，可以分解成：</p>
          <div class="equation">\[
            L_{\mathrm{vlb}}
            =
            L_T+\sum_{t=2}^{T}L_{t-1}+L_0.
          \]</div>
          <p>这三个部分分别有不同含义。</p>
          <div class="equation">\[
            L_T=
            D_{\mathrm{KL}}\left(q(x_T|x_0)\,\|\,p(x_T)\right).
          \]</div>
          <p>\(L_T\) 要求最后一步的前向分布接近标准高斯。如果 \(T\) 足够大且 schedule 合理，\(q(x_T|x_0)\) 已经非常接近 \(\mathcal{N}(0,I)\)，这一项通常没有可学习参数。</p>
          <div class="equation">\[
            L_{t-1}=
            D_{\mathrm{KL}}
            \left(
            q(x_{t-1}|x_t,x_0)
            \,\|\,
            p_\theta(x_{t-1}|x_t)
            \right),
            \qquad t=2,\ldots,T.
          \]</div>
          <p>这是最核心的项。它要求模型反向一步 \(p_\theta(x_{t-1}|x_t)\) 接近真实后验 \(q(x_{t-1}|x_t,x_0)\)。前面我们推导真实后验，就是为了知道这个 KL 项应该匹配什么。</p>
          <div class="equation">\[
            L_0=-\log p_\theta(x_0|x_1).
          \]</div>
          <p>\(L_0\) 负责最后从轻微噪声 \(x_1\) 重建 \(x_0\)。在图像模型里，它和离散像素似然或重建误差有关。</p>
          <p>如果 \(p_\theta(x_{t-1}|x_t)\) 和 \(q(x_{t-1}|x_t,x_0)\) 都是高斯，并且方差固定，那么两个高斯的 KL 主要变成均值之间的平方误差：</p>
          <div class="equation">\[
            L_{t-1}
            =
            \mathbb{E}_q
            \left[
            \frac{1}{2\sigma_t^2}
            \left\|
            \tilde{\mu}_t(x_t,x_0)-\mu_\theta(x_t,t)
            \right\|^2
            \right]
            +\mathrm{const}.
          \]</div>
          <p>这里的常数不依赖 \(\theta\)，训练时可以忽略。接下来用噪声预测参数化 \(\mu_\theta\)。因为 \(\hat{x}_0\) 可以由 \(\epsilon_\theta\) 反解出来，\(\mu_\theta\) 也就可以由 \(\epsilon_\theta\) 表示。代入后，均值误差会变成噪声误差的加权形式：</p>
          <div class="equation">\[
            L_{t-1}
            =
            \mathbb{E}_{x_0,\epsilon}
            \left[
            w_t
            \left\|
            \epsilon-\epsilon_\theta(x_t,t)
            \right\|^2
            \right]
            +\mathrm{const}.
          \]</div>
          <p>\(w_t\) 是只依赖 schedule 的权重。DDPM 论文中的 \(L_{\mathrm{simple}}\) 进一步去掉这个权重，直接训练：</p>
          <div class="equation">\[
            L_{\mathrm{simple}}=
            \mathbb{E}_{x_0,t,\epsilon}
            \left[
            \left\|
            \epsilon-\epsilon_\theta(x_t,t)
            \right\|^2
            \right].
          \]</div>
          <p>初学者读论文时最容易卡在这里：为什么突然从 likelihood 变成预测噪声？中间桥梁就是：likelihood 下界 → 反向高斯 KL → 均值匹配 → 用 \(\epsilon_\theta\) 参数化均值 → MSE。</p>
          <div class="paper-note">如果你想读 DDPM 原论文，可以先把所有 \(L_T,L_{t-1},L_0\) 都标出来，再逐个问：这一项有没有参数？它要求哪个分布接近哪个分布？它最后会不会变成一个可训练的 MSE？这样论文里的推导会清楚很多。</div>
        </section>

        <section class="article-section">
          <h2>10. 采样：DDPM 与 DDIM</h2>
          <p>训练完成后，DDPM 采样从 \(x_T\sim\mathcal{N}(0,I)\) 开始。对 \(t=T,T-1,\ldots,1\)：</p>
          <ol>
            <li>用网络预测 \(\epsilon_\theta(x_t,t)\)。</li>
            <li>由预测噪声得到 \(\hat{x}_0\)。</li>
            <li>计算反向均值 \(\mu_\theta(x_t,t)\)。</li>
            <li>加入适当方差的高斯噪声，得到 \(x_{t-1}\)。</li>
          </ol>
          <p>DDPM 的反向过程是随机的 Markov 链。DDIM 则构造了一个可以确定性采样的路径。它保持相同的边缘 \(q(x_t|x_0)\)，但不要求反向过程必须是原来的随机 Markov 链。当 \(\eta=0\) 时，DDIM 采样是确定性的；同一个初始噪声会得到一致的生成结果。这也解释了为什么 DDIM 可以用于 latent interpolation。</p>
          <p>DDIM 的一步更新常写成下面的形式。先由网络得到：</p>
          <div class="equation">\[
            \hat{x}_0=
            \frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t,t)}
            {\sqrt{\bar{\alpha}_t}}.
          \]</div>
          <p>然后从 \(t\) 走到较小的时间 \(s\)：</p>
          <div class="equation">\[
            x_s=
            \sqrt{\bar{\alpha}_s}\hat{x}_0
            +
            \sqrt{1-\bar{\alpha}_s-\sigma_{t\rightarrow s}^2}\,
            \epsilon_\theta(x_t,t)
            +
            \sigma_{t\rightarrow s}z,
            \qquad z\sim\mathcal{N}(0,I).
          \]</div>
          <p>当 \(\sigma_{t\rightarrow s}=0\) 时，这一步没有额外随机噪声，就是确定性更新；当 \(\sigma_{t\rightarrow s}\) 取合适值时，又可以恢复类似 DDPM 的随机性。DDIM 的重要意义是：训练时仍然用 DDPM 的噪声预测目标，但采样时可以跳步，用更少步数完成生成。这打开了后续各种快速采样器的大门。</p>
          <p>从学习角度看，DDIM 还揭示了一件事：模型真正学到的不是某条唯一的随机链，而是每个噪声等级上的方向信息。只要新的采样路径使用这些方向信息，并且边缘分布设计得合理，就可能生成好样本。</p>
        </section>

        <section class="article-section">
          <h2>11. SDE 与 ODE：连续时间视角</h2>
          <p>当时间步足够细时，离散扩散过程可以写成随机微分方程：</p>
          <div class="equation">\[
            dx=f(x,t)dt+g(t)dw.
          \]</div>
          <p>这里 \(f(x,t)\) 是 drift，表示确定性趋势；\(g(t)dw\) 是随机噪声项。前向 SDE 把数据变成噪声。反向 SDE 需要 score：</p>
          <div class="equation">\[
            dx=\left[f(x,t)-g(t)^2\nabla_x\log p_t(x)\right]dt+g(t)d\bar{w}.
          \]</div>
          <p>这条公式的核心含义是：如果知道每个时间 \(t\) 的 score，就能从噪声分布反向走回数据分布。</p>
          <p>同一个边缘分布族还对应一个 probability flow ODE：</p>
          <div class="equation">\[
            dx=\left[f(x,t)-\frac{1}{2}g(t)^2\nabla_x\log p_t(x)\right]dt.
          \]</div>
          <p>它没有随机项，但在每个时间点拥有和 SDE 相同的边缘分布 \(p_t(x)\)。这解释了为什么可以用确定性 ODE 采样器生成样本，也解释了 DDIM 与 probability flow 之间的联系。</p>

          <h3>VP SDE：DDPM 的连续版本</h3>
          <p>DDPM 常对应 variance preserving SDE，也就是 VP SDE：</p>
          <div class="equation">\[
            dx=-\frac{1}{2}\beta(t)x\,dt+\sqrt{\beta(t)}\,dw.
          \]</div>
          <p>它叫 variance preserving，是因为在合适条件下，过程不会让总体尺度无限变大。这个式子和离散 DDPM 的 \(x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{\beta_t}\epsilon_t\) 是同一种思想的连续极限：第一项让信号逐渐衰减，第二项加入噪声。</p>
          <p>VP SDE 的边缘分布仍有类似闭式：</p>
          <div class="equation">\[
            x(t)=\alpha(t)x(0)+\sigma(t)\epsilon,
          \]</div>
          <p>其中 \(\alpha(t)\) 随时间下降，\(\sigma(t)\) 随时间上升。离散 DDPM 的 \(\sqrt{\bar{\alpha}_t}\) 和 \(\sqrt{1-\bar{\alpha}_t}\)，就是这个连续形式的离散对应。</p>

          <h3>为什么反向 SDE 里会出现 score？</h3>
          <p>直觉上，前向 SDE 不断加入噪声，使分布越来越平滑。反向时，如果只把时间倒过来，会缺少“往数据高密度区域聚拢”的信息。score 正好提供这个信息：它告诉当前 \(x\) 在分布 \(p_t\) 下往哪里移动会让 log density 增大。</p>
          <p>因此反向漂移项不是简单的 \(-f(x,t)\)，而要额外减去 \(g(t)^2\nabla_x\log p_t(x)\)。这个项会抵消扩散造成的概率流失，把样本推回数据结构更强的位置。</p>

          <h3>连续性方程：ODE 和 Flow Matching 的公共语言</h3>
          <p>如果粒子按照速度场 \(v_t(x)\) 运动：</p>
          <div class="equation">\[
            \frac{dx}{dt}=v_t(x),
          \]</div>
          <p>那么它们形成的概率密度 \(p_t(x)\) 会满足连续性方程：</p>
          <div class="equation">\[
            \frac{\partial p_t(x)}{\partial t}
            =
            -\nabla\cdot\left(p_t(x)v_t(x)\right).
          \]</div>
          <p>这个式子的意思很朴素：一个区域里的概率质量会因为速度场流入而增加，因为速度场流出而减少。probability flow ODE 和 flow matching 都可以用这句话理解，只是它们得到速度场的方式不同。</p>
        </section>

        <section class="article-section">
          <h2>12. Flow Matching：直接学习速度场</h2>
          <p>Flow Matching 从另一个角度看生成：我们不一定先学习 score，再由 score 构造反向 SDE/ODE；也可以直接学习一个速度场：</p>
          <div class="equation">\[
            \frac{dx_t}{dt}=v_\theta(x_t,t).
          \]</div>
          <p>最简单的路径是把噪声样本 \(x_0\) 和数据样本 \(x_1\) 线性插值：</p>
          <div class="equation">\[
            x_t=(1-t)x_0+tx_1.
          \]</div>
          <p>这条路径的速度是：</p>
          <div class="equation">\[
            u_t=\frac{dx_t}{dt}=x_1-x_0.
          \]</div>
          <p>训练时让网络回归这个速度：</p>
          <div class="equation">\[
            L(\theta)=\mathbb{E}\left[\|v_\theta(x_t,t)-u_t\|^2\right].
          \]</div>
          <p>更一般的 Flow Matching 使用不同 probability path 和条件速度场。这里要区分两个概念。</p>
          <p><strong>条件路径</strong> \(p_t(x|z)\)：给定某个条件变量 \(z\)，比如一对噪声和数据样本，定义一条从源分布到目标分布的路径。</p>
          <p><strong>边缘路径</strong> \(p_t(x)\)：把所有条件变量积分掉后，真实样本总体在时间 \(t\) 的分布。</p>
          <div class="equation">\[
            p_t(x)=\int p_t(x|z)p(z)dz.
          \]</div>
          <p>训练时我们通常知道条件速度 \(u_t(x|z)\)，但生成时只能看到 \(x_t\)，不知道它对应哪个 \(z\)。Flow Matching 的关键结果是：直接回归条件速度，在期望意义下可以学到边缘速度场：</p>
          <div class="equation">\[
            v_t(x)=\mathbb{E}\left[u_t(x|z)\mid x_t=x\right].
          \]</div>
          <p>这和 denoising score matching 的味道很像：训练监督来自条件对象，但模型最终学到的是边缘对象。Diffusion 里是条件 score 到边缘 score；Flow Matching 里是条件速度到边缘速度。</p>
          <p>DiffusionFlow 的文章强调：在高斯源分布和合适 schedule 下，diffusion 与 flow matching 可以相互转换。它们常常不是“谁替代谁”的关系，而是同一类分布运输思想的不同参数化。</p>

          <h3>Flow Matching 和 Diffusion 的区别在哪里？</h3>
          <p>Diffusion 的经典训练目标经常从“加噪和去噪”出发，学习 score 或噪声；Flow Matching 则从“概率质量如何被速度场运输”出发，直接学习 ODE 速度。二者都可以从简单分布生成复杂分布，但思考入口不同。</p>
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>问题</th><th>Diffusion/Score 视角</th><th>Flow Matching 视角</th></tr>
              </thead>
              <tbody>
                <tr><td>路径怎么来？</td><td>先设计加噪过程，让数据逐渐变成高斯。</td><td>直接设计从源分布到数据分布的 probability path。</td></tr>
                <tr><td>网络学什么？</td><td>噪声、score、\(x_0\) 或 \(v\) 参数化。</td><td>速度场 \(v_t(x)\)。</td></tr>
                <tr><td>生成怎么做？</td><td>反向 SDE、probability flow ODE 或各种离散采样器。</td><td>求解 ODE \(\frac{dx}{dt}=v_\theta(x,t)\)。</td></tr>
                <tr><td>核心方程</td><td>反向 SDE / score identity。</td><td>连续性方程。</td></tr>
              </tbody>
            </table>
          </div>

        </section>

        <section class="article-section">
          <h2>13. 几种模型输出的统一理解</h2>
          <p>论文里常见不同输出：预测噪声 \(\epsilon\)、预测干净样本 \(x_0\)、预测 score、预测 velocity \(v\)、预测 flow \(u\)。它们不是互不相干的概念，而是同一个 noisy sample 的不同坐标表达。</p>
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>输出</th><th>含义</th><th>优点</th><th>常见位置</th></tr>
              </thead>
              <tbody>
                <tr><td>\(\epsilon\)</td><td>加入到 \(x_0\) 的噪声</td><td>DDPM 中训练稳定，目标简单</td><td>DDPM, Stable Diffusion 早期实现</td></tr>
                <tr><td>\(x_0\)</td><td>估计原始干净数据</td><td>直观，便于解释 denoiser</td><td>DDIM 推导、denoiser 表达</td></tr>
                <tr><td>score</td><td>\(\nabla_x\log p_t(x)\)</td><td>直接给出概率密度上升方向</td><td>Score-based SDE</td></tr>
                <tr><td>\(v\)</td><td>\(\alpha_t\epsilon-\sigma_t x_0\) 一类混合参数化</td><td>平衡高噪声和低噪声区域的数值稳定性</td><td>Imagen, Stable Diffusion v-parameterization</td></tr>
                <tr><td>\(u\)</td><td>Flow Matching 速度场</td><td>直接学习分布运输方向</td><td>Flow Matching, Rectified Flow</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        <section class="article-section">
          <h2>14. 初学者读论文时应该抓住什么？</h2>
          <p>读 DDPM 论文时，先抓住三件事：前向过程是固定高斯链；反向过程用神经网络参数化；训练目标从 VLB 简化成噪声 MSE。</p>
          <p>读 DDIM 论文时，关注它如何保持相同训练边缘分布，却改变采样路径，从而实现确定性和跳步采样。</p>
          <p>读 score-based SDE 论文时，关注 score 为什么能定义反向 SDE，以及 probability flow ODE 为什么能和 SDE 共享边缘分布。</p>
          <p>读 flow matching 论文时，关注 probability path、conditional vector field、marginal vector field 这三个概念，以及为什么回归速度场可以训练生成模型。</p>
          <p>如果读论文时迷路，可以反复回到这一句：生成模型是在学习一条从简单分布到复杂数据分布的运输路径。Diffusion、Score、ODE、Flow Matching 只是描述和学习这条路径的不同语言。</p>
        </section>

        <section class="article-section">
          <h2>15. 论文符号对照：看到不同记号不要慌</h2>
          <p>不同论文的符号经常不一样，但数学对象往往是同一个。读论文时先把符号翻译成本文的几个核心对象，会轻松很多。</p>
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>你在论文里看到</th><th>常见含义</th><th>和本文的关系</th><th>读法</th></tr>
              </thead>
              <tbody>
                <tr><td>\(\sigma\)</td><td>噪声标准差</td><td>类似 \(\sqrt{1-\bar{\alpha}_t}\)</td><td>表示当前样本有多脏。</td></tr>
                <tr><td>\(\alpha(t),\sigma(t)\)</td><td>连续时间的信号系数和噪声系数</td><td>\(x_t=\alpha(t)x_0+\sigma(t)\epsilon\)</td><td>DDPM 闭式公式的连续版本。</td></tr>
                <tr><td>\(s_\theta(x,t)\)</td><td>score 网络</td><td>\(s_\theta\approx\nabla_x\log p_t(x)\)</td><td>告诉样本往哪里更像数据。</td></tr>
                <tr><td>\(D_\theta(x,\sigma)\)</td><td>denoiser</td><td>预测 \(x_0\) 的网络</td><td>EDM 等文章喜欢从去噪器角度写。</td></tr>
                <tr><td>\(\epsilon_\theta(x,t)\)</td><td>噪声预测网络</td><td>DDPM 的常用参数化</td><td>预测当初加进去的噪声。</td></tr>
                <tr><td>\(v_\theta(x,t)\)</td><td>速度或 velocity 参数化</td><td>可能是 diffusion 的 \(v\)-prediction，也可能是 flow 的速度场</td><td>要看上下文：是线性组合目标，还是 ODE 速度。</td></tr>
                <tr><td>probability flow ODE</td><td>和反向 SDE 共享边缘分布的确定性 ODE</td><td>由 score 构造速度</td><td>解释为什么不用随机噪声也能采样。</td></tr>
                <tr><td>conditional flow matching</td><td>用条件路径监督速度场</td><td>条件速度回归到边缘速度</td><td>类似“用条件信息训练，生成时只看当前点”。</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        <section class="article-section">
          <h2>16. 常见误解与纠正</h2>
          <h3>误解一：Diffusion 就是给图片加噪再去噪</h3>
          <p>这句话只说对了一半。加噪和去噪是训练和采样形式，但更本质的说法是：Diffusion 学习一族随时间变化的概率分布，以及从高噪声分布回到数据分布的方向。只把它理解成图像去噪，会很难理解 score、SDE、ODE 和 flow matching。</p>

          <h3>误解二：网络预测噪声，所以它只是学会了找噪声</h3>
          <p>预测噪声等价于预测当前 noisy sample 相对于干净数据的偏移方向，也等价于在已知尺度下预测条件 score。模型不是只在识别随机花纹，而是在所有噪声等级上学习“如何把样本拉回数据流形”。</p>

          <h3>误解三：DDPM 的反向过程就是把前向公式倒过来</h3>
          <p>前向公式可以从 \(x_0\) 得到 \(x_t\)，但反过来从 \(x_t\) 得到 \(x_0\) 是不适定问题。许多 \(x_0\) 都可能加噪到类似的 \(x_t\)。所以反向过程必须学习条件分布，而不是简单代数求逆。</p>

          <h3>误解四：SDE/ODE 很高级，可以先完全跳过</h3>
          <p>训练第一个 DDPM 时确实可以先不深究 SDE。但如果想读后续论文，比如 score-based generative modeling、EDM、flow matching、rectified flow、consistency model，连续时间语言几乎无法绕开。好消息是，它们的核心直觉仍然是本文这条路径：数据分布和噪声分布之间的运输。</p>

          <h3>误解五：Flow Matching 是 Diffusion 的完全替代品</h3>
          <p>更好的理解是：Flow Matching 和 Diffusion 都在描述从简单分布到数据分布的路径。Diffusion 常从 score 和反向随机过程出发；Flow Matching 常从速度场和连续性方程出发。它们有区别，也有可转换的部分。读新论文时，与其问“谁替代谁”，不如问“这篇论文选择了怎样的路径、怎样的参数化、怎样的训练监督”。</p>
        </section>

        <section class="article-section">
          <h2>17. 学完本文后，如何真正开始读论文？</h2>
          <p>推荐顺序不是按发表时间死排，而是按概念依赖来排。</p>
          <ol>
            <li><strong>DDPM。</strong>重点看前向闭式、后验高斯、VLB 分解、\(L_{\mathrm{simple}}\)。如果这些懂了，Diffusion 的离散基础就稳了。</li>
            <li><strong>DDIM。</strong>重点看为什么可以换采样路径，以及确定性采样如何从 \(\hat{x}_0\) 和 \(\epsilon_\theta\) 构造。</li>
            <li><strong>Score SDE。</strong>重点看 score、reverse-time SDE、probability flow ODE。不要被随机微分方程符号吓住，先抓住“score 修正反向漂移”。</li>
            <li><strong>EDM 或其他采样器文章。</strong>重点看噪声参数化、预条件、schedule 和数值求解器。它们经常是在改善“同一条路径怎么走得更稳更快”。</li>
            <li><strong>Flow Matching / Rectified Flow。</strong>重点看 probability path、velocity field、continuity equation，以及条件速度为什么能训练边缘速度。</li>
          </ol>
          <p>每读一篇论文，都可以用下面四个问题检查理解：</p>
          <ol>
            <li>这篇论文的数据端和噪声端分别是什么分布？</li>
            <li>它选择了什么路径连接这两个分布？</li>
            <li>神经网络输出的是什么对象：噪声、score、\(x_0\)、速度，还是别的参数化？</li>
            <li>训练损失为什么能监督这个对象？采样时又如何使用它？</li>
          </ol>
          <div class="insight-box">如果这四个问题都能回答，说明你已经不是在背公式，而是在理解生成模型的结构。</div>
        </section>

        <section class="article-section">
          <h2>18. 一个二维玩具例子：从数据到生成完整走一遍</h2>
          <p>为了确认我们不是只在符号层面理解 Diffusion，现在用一个二维玩具例子把整个流程走一遍。假设真实数据不是图片，而是二维平面上的点。真实分布由两个簇组成：左边一团点，右边一团点。你可以把它想成最简单的“数据流形”：真实样本大多落在两个区域，平面其他地方概率很低。</p>
          <p>训练集里每个样本是 \(x_0=(x^{(1)},x^{(2)})\)。我们并不知道真实密度 \(p_{\mathrm{data}}(x)\) 的解析式，只知道手上有一批样本。生成模型的目标是：以后不再依赖训练集，也能采样出新的点，而且这些点应该落在两个簇附近。</p>
          <p>第一步，定义前向加噪。对每个真实点 \(x_0\)，随机选一个时间 \(t\)，采样一份标准高斯噪声 \(\epsilon\)，然后合成：</p>
          <div class="equation">\[
            x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon.
          \]</div>
          <p>当 \(t\) 很小时，点仍然靠近原来的簇，只是有一点抖动；当 \(t\) 增大，两个簇开始变模糊；当 \(t\) 接近 \(T\)，所有点几乎都混成一个标准高斯云。这个过程不是模型学出来的，而是我们人为制造的训练题目。</p>
          <p>第二步，构造监督信号。注意我们在训练时知道 \(\epsilon\)，因为它就是刚刚采样出来的噪声。所以可以让网络看 \(x_t\) 和 \(t\)，输出 \(\epsilon_\theta(x_t,t)\)，并用真实 \(\epsilon\) 监督它。这个任务在二维例子里非常直观：如果一个点还在左边簇附近，网络要判断它偏离簇中心的噪声方向；如果一个点在两个簇之间，网络要根据统计规律判断它更可能来自哪个簇。</p>
          <p>第三步，理解网络学到的东西。低噪声时，两个簇还分得很清楚，模型学的是局部去噪：把点拉回附近簇。中等噪声时，两个簇开始重叠，模型学的是概率判断：同一个 \(x_t\) 可能来自左簇，也可能来自右簇，网络输出的是平均意义上的方向。高噪声时，几乎看不出原始簇结构，模型学的是从大范围噪声云中慢慢建立全局结构。</p>
          <p>第四步，生成。采样时我们不再有 \(x_0\)，只从 \(x_T\sim\mathcal{N}(0,I)\) 开始。网络根据当前 \(x_t\) 预测噪声，再用反向公式得到 \(x_{t-1}\)。一开始，点云只是无结构高斯；走几步后，点云开始往两个区域分裂；继续走，两个簇越来越清晰；最后得到的新点落在真实数据簇附近。</p>
          <p>这个二维例子可以解释为什么 Diffusion 适合高维图像。图像虽然复杂得多，但逻辑相同：真实图像分布是高维空间里有结构的区域；前向加噪逐渐抹掉结构；训练让模型在每个噪声等级上学习恢复结构的方向；采样从纯噪声开始，逐级把结构长出来。</p>
          <div class="insight-box">如果你能在脑子里想象二维点云从两个簇变成高斯云，再从高斯云分裂回两个簇，那么你已经抓住了 Diffusion 的主干。图像模型只是把这个二维直觉搬到极高维空间，并用 U-Net 或 Transformer 表达更复杂的方向场。</div>
        </section>

        <section class="article-section">
          <h2>19. 把 DDPM 论文的核心推导压成一条线</h2>
          <p>现在我们从论文视角重新串一次。这样做的目的不是重复，而是帮助你形成读论文时的“骨架”。以后看到新论文，你可以先判断它是在这条线的哪一步做改造。</p>
          <p><strong>第一步：目标是最大似然。</strong>我们希望模型给真实数据高概率，也就是最大化 \(\log p_\theta(x_0)\)。如果能直接计算这个量，就可以像普通概率模型一样训练。但 DDPM 的生成过程包含很多隐变量 \(x_1,\ldots,x_T\)，所以直接边缘化很难。</p>
          <div class="equation">\[
            p_\theta(x_0)=\int p_\theta(x_{0:T})dx_{1:T}.
          \]</div>
          <p><strong>第二步：引入一个已知的前向过程。</strong>我们定义 \(q(x_{1:T}|x_0)\)，它负责把数据加噪成高斯。这个过程不是要生成样本，而是为了构造一个可以训练的下界。它的好处是所有条件分布都是高斯，很多量可以解析计算。</p>
          <p><strong>第三步：用 Jensen 不等式得到 ELBO。</strong>把 \(q\) 乘进去再除出来，得到一个期望形式，然后把 \(\log\) 放进期望内部，就得到下界。这个步骤的意义是：我们把难算的 log marginal likelihood 变成了一个可以采样估计、可以分解的目标。</p>
          <p><strong>第四步：分解下界。</strong>下界分成三类项：最后的 \(x_T\) 是否像标准高斯，中间每一步反向分布是否接近真实后验，最后从 \(x_1\) 重建 \(x_0\) 是否合理。其中最重要的是中间 KL：</p>
          <div class="equation">\[
            D_{\mathrm{KL}}
            \left(
            q(x_{t-1}|x_t,x_0)
            \,\|\,
            p_\theta(x_{t-1}|x_t)
            \right).
          \]</div>
          <p><strong>第五步：真实后验可以算。</strong>因为 \(q(x_t|x_{t-1})\) 和 \(q(x_{t-1}|x_0)\) 都是高斯，它们相乘仍然是高斯，所以 \(q(x_{t-1}|x_t,x_0)\) 可以精确写出均值和方差。这个地方是 DDPM 推导里最关键的代数部分。</p>
          <p><strong>第六步：模型后验也设成高斯。</strong>我们令：</p>
          <div class="equation">\[
            p_\theta(x_{t-1}|x_t)=
            \mathcal{N}(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t)).
          \]</div>
          <p>如果方差固定，训练主要就是让 \(\mu_\theta\) 接近真实后验均值 \(\tilde{\mu}_t\)。两个高斯的 KL 于是变成均值之间的平方误差。</p>
          <p><strong>第七步：用噪声预测参数化均值。</strong>真实后验均值依赖 \(x_0\)，但采样时我们没有 \(x_0\)。于是网络预测 \(\epsilon\)，再由 \(\epsilon_\theta\) 反推出 \(\hat{x}_0\)，再代入均值公式。这样模型只需要学一个统一任务：给定 \(x_t\) 和 \(t\)，预测当初加入的噪声。</p>
          <p><strong>第八步：得到简单 MSE。</strong>经过上述代换，复杂的 VLB 中间项变成带权的噪声 MSE。实践中进一步简化，去掉权重，得到 \(L_{\mathrm{simple}}\)。这就是为什么代码里看起来只有一行 MSE，但它背后其实连着最大似然、变分下界、后验高斯和均值参数化。</p>
          <p>所以，DDPM 不是“凭直觉训练一个去噪器”这么简单。它的理论闭环是完整的：概率模型定义了反向生成链；前向过程提供变分分布；ELBO 给出训练目标；高斯代数让目标可计算；噪声预测让训练稳定而简单。</p>
        </section>

        <section class="article-section">
          <h2>20. 图像模型里的网络结构：它不改变概率论，只负责表达函数</h2>
          <p>很多教程会很快跳到 U-Net、attention、time embedding、classifier-free guidance，这会让初学者误以为 Diffusion 的核心是某种神经网络结构。其实概率论部分和网络结构部分要分开看。</p>
          <p>从理论上说，网络只是在近似一个函数。这个函数可能是 \(\epsilon_\theta(x_t,t)\)，也可能是 \(s_\theta(x_t,t)\)、\(x_{\theta,0}(x_t,t)\) 或 \(v_\theta(x_t,t)\)。无论用 U-Net、DiT 还是其他架构，它们都在回答同一个问题：当前噪声样本 \(x_t\) 和噪声等级 \(t\) 给定后，应该输出什么方向信息。</p>
          <p>为什么图像 Diffusion 常用 U-Net？因为 U-Net 适合处理图像的多尺度结构。低层特征关心边缘和纹理，高层特征关心物体和语义；跳连可以把细节信息传回来。时间步 \(t\) 通常通过 time embedding 注入网络，让同一个网络知道当前面对的是低噪声、中噪声还是高噪声。</p>
          <p>attention 的作用是让远距离区域交换信息。生成图像时，一个物体的局部纹理要和整体布局协调，单纯卷积的局部感受野可能不够。加入 self-attention 或 cross-attention 后，模型可以更好地处理全局关系。文本到图像模型中的 cross-attention，则把文字条件注入图像生成过程。</p>
          <p>但是，网络结构再复杂，也没有改变本文推导出的核心关系。训练时仍然是采样 \(t\)、合成 \(x_t\)、预测目标、计算损失；采样时仍然是从噪声开始，反复调用网络得到方向，再更新样本。理解这一点很重要：架构是表达能力，概率路径是生成逻辑。</p>
          <div class="paper-note">读 Stable Diffusion 或 Latent Diffusion 时，也可以用同样框架理解。区别是扩散不直接发生在像素空间，而发生在 VAE 编码后的 latent space。这样计算更便宜，但前向加噪、噪声预测、反向采样这些理论骨架仍然存在。</div>
        </section>

        <section class="article-section">
          <h2>21. 多轮自检：你是否真的能继续读论文了？</h2>
          <p>最后不只做一次检查，而是从三个层次反复问自己：定义是否清楚，推导是否能复现，读论文时是否知道作者在改什么。每个问题后面都给出回看位置。如果某个问题答不上来，不是失败，而是说明需要回到对应章节重新推一遍。</p>

          <h3>第一轮：基本对象有没有混淆？</h3>
          <p><strong>检查一：你能否解释 \(x_0,x_t,x_T,\epsilon,\epsilon_\theta\) 分别是什么？</strong>\(x_0\) 是干净数据，\(x_t\) 是第 \(t\) 个噪声等级的样本，\(x_T\) 应该接近标准高斯，\(\epsilon\) 是真实采样噪声，\(\epsilon_\theta\) 是网络预测噪声。卡住时回到第 0 节符号表和第 3 节。</p>
          <p><strong>检查二：你能否解释为什么 \(q\) 不是模型、\(p_\theta\) 才是模型？</strong>\(q\) 是人为规定的前向加噪过程，它不学习；\(p_\theta\) 是反向生成过程，需要神经网络学习。卡住时回到第 5 节。</p>
          <p><strong>检查三：你能否解释为什么 \(q(x_t|x_0)\) 可以一步采样？</strong>回答时应该说出独立高斯线性组合仍是高斯，信号系数累计成 \(\sqrt{\bar{\alpha}_t}\)，噪声方差累计成 \(1-\bar{\alpha}_t\)。如果只能背公式，还不算真正理解。卡住时回到第 2.5 节和第 4 节。</p>

          <h3>第二轮：核心推导能不能自己走完？</h3>
          <p><strong>检查四：你能否解释为什么 \(q(x_{t-1}|x_t,x_0)\) 是高斯？</strong>回答时应该说出贝叶斯公式、Markov 性、两个高斯密度相乘、二次型配方、精度相加。这里是读 DDPM 论文最关键的门槛。卡住时回到第 6 节。</p>
          <p><strong>检查五：你能否解释为什么预测噪声可以训练生成模型？</strong>回答时不能只说“DDPM 就是这么做的”，而要说清楚：预测噪声可以反推出 \(\hat{x}_0\)，进而参数化反向均值；反向高斯 KL 在方差固定时变成均值 MSE；均值 MSE 又可以写成噪声 MSE。卡住时回到第 7 节和第 9 节。</p>
          <p><strong>检查六：你能否把 score 和噪声预测联系起来？</strong>应该能写出：</p>
          <div class="equation">\[
            \nabla_{x_t}\log q(x_t|x_0)
            =
            -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}.
          \]</div>
          <p>并且能解释这只是条件 score，真正生成需要边缘 score，而 denoising score matching 正是用条件监督学习边缘方向。卡住时回到第 8 节。</p>

          <h3>第三轮：能不能带着框架读新论文？</h3>
          <p><strong>检查七：你能否解释 SDE、ODE、Flow Matching 的共同点？</strong>它们都在描述概率分布如何从噪声端移动到数据端。SDE 用随机过程和 score 修正反向漂移，probability flow ODE 用确定性速度保持同样边缘分布，Flow Matching 直接回归满足连续性方程的速度场。卡住时回到第 11 节和第 12 节。</p>
          <p><strong>检查八：你能否看到一篇新论文时，立刻定位它改变的是哪一层？</strong>有些论文改变噪声 schedule，有些改变网络输出参数化，有些改变采样器，有些把像素空间换成 latent space，有些把 score 语言换成 flow 语言。只要能把创新点放回“路径、目标、参数化、采样”这四个位置，就不会被新名词牵着走。卡住时回到第 14 节、第 15 节和第 17 节。</p>
          <p><strong>检查九：你能否把公式翻译成口头解释？</strong>比如 \(\bar{\alpha}_t\) 不是一个神秘符号，而是原始信号保留比例；\(\tilde{\mu}_t\) 不是凭空出现的均值，而是由 \(x_t\) 和 \(x_0\) 两个信息源按可靠性加权；score 不是抽象梯度，而是往高概率区域移动的方向。能做到这一点，说明公式已经变成理解，而不是记忆负担。卡住时回到第 4.5 节、第 6 节和第 8 节。</p>
          <p><strong>检查十：你能否解释为什么采样慢、为什么可以加速？</strong>慢是因为模型沿着很多离散噪声等级逐步移动，每一步只做局部修正；加速则来自更好的时间步选择、更稳定的 ODE/SDE 求解器、或者更直接的路径学习。这样看 DDIM、DPM-Solver、EDM、consistency model 时，就能明白它们大多是在改进“怎么走这条路”。卡住时回到第 10 节、第 11 节和第 12 节。</p>
          <p>如果这些问题还不能顺畅回答，就回到对应章节重新推一遍。Diffusion 理论最怕“看着都懂，合上全忘”。最有效的学习方式是自己在纸上从 \(x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon\) 开始，一路推到训练损失，再把它翻译成采样算法。真正掌握后，你应该能把每个公式都说成人话，也能把每句直觉重新写回公式。等你再读新论文时，就会知道作者究竟是在改路径、改目标、改采样器，还是改网络表达；这时阅读会从被动接受变成主动拆解，并形成自己的判断。</p>
        </section>

        <section class="article-section references">
          <h2>参考资料</h2>
          <ul>
            <li><a href="https://arxiv.org/abs/2006.11239" target="_blank" rel="noreferrer">Ho, Jain, Abbeel, Denoising Diffusion Probabilistic Models, 2020</a></li>
            <li><a href="https://arxiv.org/abs/2010.02502" target="_blank" rel="noreferrer">Song, Meng, Ermon, Denoising Diffusion Implicit Models, 2020</a></li>
            <li><a href="https://arxiv.org/abs/2011.13456" target="_blank" rel="noreferrer">Song et al., Score-Based Generative Modeling through Stochastic Differential Equations, 2021</a></li>
            <li><a href="https://arxiv.org/abs/2102.09672" target="_blank" rel="noreferrer">Nichol, Dhariwal, Improved Denoising Diffusion Probabilistic Models, 2021</a></li>
            <li><a href="https://arxiv.org/abs/2206.00364" target="_blank" rel="noreferrer">Karras et al., Elucidating the Design Space of Diffusion-Based Generative Models, 2022</a></li>
            <li><a href="https://arxiv.org/abs/2207.12598" target="_blank" rel="noreferrer">Ho, Salimans, Classifier-Free Diffusion Guidance, 2022</a></li>
            <li><a href="https://arxiv.org/abs/2210.02747" target="_blank" rel="noreferrer">Lipman et al., Flow Matching for Generative Modeling, 2022</a></li>
            <li><a href="https://arxiv.org/abs/2112.10752" target="_blank" rel="noreferrer">Rombach et al., High-Resolution Image Synthesis with Latent Diffusion Models, 2021</a></li>
            <li><a href="https://diffusionflow.github.io/" target="_blank" rel="noreferrer">Diffusion Meets Flow Matching</a></li>
            <li><a href="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/" target="_blank" rel="noreferrer">Lilian Weng, What are Diffusion Models?</a></li>
            <li><a href="https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html" target="_blank" rel="noreferrer">Cambridge MLG Blog, Flow Matching Guide and Code</a></li>
            <li><a href="https://arxiv.org/abs/2506.02070" target="_blank" rel="noreferrer">arXiv:2506.02070</a></li>
          </ul>
        </section>
      `,
    },
  ],
  tags: [
    "Diffusion Model",
    "Generative Model",
    "Machine Learning",
    "Probability",
    "Maximum Likelihood",
    "KL Divergence",
    "DDPM",
    "Score Matching",
    "Flow Matching",
    "SDE",
    "ODE"
  ],
  categories: [{ name: "学习笔记", count: 2 }],
};
