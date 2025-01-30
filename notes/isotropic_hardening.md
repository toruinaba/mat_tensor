多軸応力下の降伏関数を以下とする.
$$
\begin{align}
f_i &= \sqrt{\frac{3}{2}\boldsymbol{S_i}:\boldsymbol{S_i}} - \sigma^i_y \\
&=\sqrt{\frac{3}{2}}||\boldsymbol{S_i}||-\sigma^i_y \\
&=q_i - \sigma^i_y
\end{align}
$$
ただし降伏応力度$\sigma^i_y$は以下とする.
$$
\sigma^i_y = \sigma_{y0} + R_{i}
$$

このとき降伏応力の等方硬化成分$R_i$についてVoce型の進展則を仮定すると
$$
\dot R_{i+1} = b\left(
	Q-R_{i+1}
\right)\tilde{\epsilon_p}
$$
ただし$\tilde{\epsilon_p}$は相当塑性ひずみ増分である.
今$i+1$における等方硬化成分$R_{i+1}$が増分型で以下のように表されるとする.
$$
\begin{align}
R_{i+1} &= R_i+\dot R_{i+1} \\
&= R_i + b(Q - R_{i+1})\tilde{\epsilon_p} \\
\therefore R_{i+1} &= \theta (R_i + bQ\tilde{\epsilon_p})
\end{align}
$$
ただし
$$
\theta = \frac{1}{1+b\tilde{\epsilon_p}}
$$
これより$i+1$における降伏応力度$\sigma^{i+1}_y$は以下のようになる.
$$
\begin{align}
\sigma^{i+1}_y &= \sigma_{y0} + R_{i+1} \\
&= \sigma_{y0} + \theta\left(R_i + bQ\tilde{\epsilon_p} \right)
\end{align}
$$
応力進展について考える.
今弾性試行ひずみ$\boldsymbol{\epsilon^{etri}_{i+1}}$を以下のように定義する.
$$
\boldsymbol{\epsilon^{etri}_{i+1}}=\boldsymbol{\epsilon_{i+1}}-\boldsymbol{\epsilon^p_i}
$$
このとき$i+1$における応力度$\boldsymbol{\sigma_{i+1}}$および弾性ひずみ$\boldsymbol{\epsilon^e_{i+1}}$は
$$
\begin{align}
\boldsymbol{\sigma_{i+1}}&=\boldsymbol{\tilde{D_e}}:\boldsymbol{\epsilon^e_{i+1}} \\
&= \boldsymbol{\tilde{D_e}}:\left(
	\boldsymbol{\epsilon_{i+1}} - \boldsymbol{\epsilon^p_{i+1}}
\right) \\
&= \boldsymbol{\tilde{D_e}}:\left(
	\boldsymbol{\epsilon_{i+1}} -
	\boldsymbol{\epsilon^p_{i}} +
	\boldsymbol{\epsilon^p_{i}} -
	\boldsymbol{\epsilon^p_{i+1}}
\right) \\
&= \boldsymbol{\tilde{D_e}}:\left(
	\boldsymbol{\epsilon_{i+1}} -
	\boldsymbol{\epsilon^p_{i}} -
	\dot{\boldsymbol{\epsilon_p}}
\right) \\
\end{align}
$$
ここで
$$
\boldsymbol{\dot\epsilon_p} = \Delta\gamma \frac{\partial f_{i+1}}{\partial \boldsymbol{S_{i+1}}}
$$
また
$$
\frac{\partial f}{\partial\boldsymbol{S_{i+1}}} = \sqrt{\frac{3}{2}}\boldsymbol{\frac{S_{i+1}}{||S_{i+1}||}} = \sqrt{\frac{3}{2}}\boldsymbol{\bar N^{i+1}_\sigma}
$$
これより
$$
\boldsymbol{\dot\epsilon_p} = \Delta\gamma\frac{\partial f_{i+1}}{\partial \boldsymbol{S_{i+1}}} =
\sqrt{\frac{3}{2}}\Delta\gamma\boldsymbol{\bar N^{i+1}_\sigma}
$$
以上から$\sigma_{i+1}$は以下のように表すことができる.
$$
\begin{align}
\boldsymbol{\sigma_{i+1}} &= \boldsymbol{\tilde{D_e}}:\left(
	\boldsymbol{\epsilon_{i+1}} -
	\boldsymbol{\epsilon^p_{i}} -
	\dot{\boldsymbol{\epsilon_p}}
\right) \\
&=\boldsymbol{\tilde{D_e}}:\boldsymbol{\epsilon^{etri}_{i+1}} -
\boldsymbol{\tilde{D_e}}:\boldsymbol{\dot\epsilon_p} \\
&=\boldsymbol{\tilde{D_e}}:\boldsymbol{\epsilon^{etri}_{i+1}} -
\sqrt{6}G\Delta\gamma\boldsymbol{\bar N^{i+1}_\sigma} \\
&=\boldsymbol{\sigma^{tri}_{i+1}} -
\sqrt{6}G\Delta\gamma\boldsymbol{\bar N^{i+1}_\sigma} \\
\end{align}
$$
また両辺偏差成分をとると
$$
\begin{align}
\boldsymbol{S_{i+1}}&=\boldsymbol{\tilde{I_d}}:\boldsymbol{\sigma^{tri}_{i+1}}-
\sqrt{6}G\Delta\gamma\boldsymbol{\bar N^{i+1}_\sigma} \\
||\boldsymbol{S_{i+1}}||\boldsymbol{\bar N^{i+1}_\sigma} &=
\boldsymbol{\tilde{I_d}}:\boldsymbol{\sigma^{tri}_{i+1}}-
\sqrt{6}G\Delta\gamma\boldsymbol{\bar N^{i+1}_\sigma} \\
\end{align}
$$
ここで
$$
q_{i+1}=\sqrt{\frac{3}{2}}||\boldsymbol{S_{i+1}}||
$$
であることに着目すると
$$
\left(
	\sqrt{\frac{2}{3}}q_{i+1}+\sqrt{6}G\Delta\gamma
\right)\boldsymbol{\bar N^{i+1}_\sigma} = \boldsymbol{\tilde{I_d}}:
	\boldsymbol{\sigma^{tri}_{i+1}}
$$
いま両辺ノルムをとり
$$
\boldsymbol{\tilde{I_d}:\boldsymbol{\sigma^{tri}_{i+1}}}=\boldsymbol{S^{tri}_{i+1}}
$$

$$
\sqrt{\frac{3}{2}}||\boldsymbol{S^{tri}_{i+1}}||=q^{tri}_{i+1}
$$
とおくと
$$
\sqrt{\frac{2}{3}}q_{i+1}+\sqrt{6}G\Delta\gamma=\sqrt{\frac{2}{3}}q^{tri}_{i+1}
$$
また降伏関数が降伏進展時に0であることを考えると
$$
q_{i+1} = \sigma_{y0} + \theta\left(R_i + bQ\Delta\gamma\right)
$$
であることからこれを代入することで下記の$\Delta\gamma$に関する非線形方程式が得られる.
$$
\sigma_{y0} + \theta\left(R_i + bQ\Delta\gamma\right)+
3G\Delta\gamma-q^{tri}_{i+1}=0
$$
よって
$$
F(\Delta\gamma)=\sigma_{y0} + \theta\left(R_i + bQ\Delta\gamma\right)+
3G\Delta\gamma-q^{tri}_{i+1}
$$
とおくと
$$
F'(\Delta\gamma)=3G+\frac{\partial \theta}{\partial \Delta\gamma} (R_i+bQ\Delta\gamma)+\theta bQ
$$
ここで
$$
\frac{\partial \theta}{\partial \Delta\gamma}=-b\theta^2
$$
を用いれば
$$
F'(\Delta\gamma)=3G-b\theta^2 (R_i+bQ\Delta\gamma)+\theta bQ
$$
以上よりニュートン法を用いて$\Delta\gamma$を計算可能である.

続いてコンシステント接線剛性演算子を考える. いま$\boldsymbol{\bar N^{i+1}_\sigma}$と$\boldsymbol{S^{tri}_{i+1}}$の関係より
$$
\boldsymbol{\bar N^{i+1}_\sigma}=\frac{\boldsymbol{S^{tri}_{i+1}}}{||\boldsymbol{S^{tri}_{i+1}}||}
$$
これより
$$
\begin{align}
\boldsymbol{\sigma_{i+1}} &=\boldsymbol{\sigma^{tri}_{i+1}} -
\sqrt{6}G\Delta\gamma\frac{\boldsymbol{S^{tri}_{i+1}}}{||\boldsymbol{S^{tri}_{i+1}}||}　\\
&=\boldsymbol{\sigma_{i+1}}\left(
	\boldsymbol{\sigma^{tri}_{i+1}(\boldsymbol{\epsilon^{etri}_{i+1}})},
	\Delta\gamma(\boldsymbol{\epsilon^{etri}_{i+1}}),
	\boldsymbol{S^{tri}_{i+1}}\left(
		\boldsymbol{\sigma^{tri}_{i+1}}(\boldsymbol{\epsilon^{etri}_{i+1}})
	\right)
\right)
\end{align}
$$

よってテンソルの微分則を適用するとコンシステント接線剛性$\boldsymbol{\tilde{D_{ep}}}$は
$$
\begin{align}
\boldsymbol{\tilde{D_{ep}}} &= \frac{\partial\boldsymbol{\sigma_{i+1}}}{\partial\boldsymbol{\epsilon^{etri}_{i+1}}} \\
	&=\frac{\partial\boldsymbol{\sigma_{i+1}}}{\partial\boldsymbol{\sigma^{tri}_{i+1}}}
	\frac{\partial\boldsymbol{\sigma^{tri}_{i+1}}}{\partial\boldsymbol{\epsilon^{etri}_{i+1}}}+
	\frac{\partial\boldsymbol{\sigma_{i+1}}}{\Delta\gamma} \otimes
	\frac{\partial\Delta\gamma}{\partial\boldsymbol{\epsilon^{etri}_{i+1}}}+
	\frac{\partial\boldsymbol{\sigma_{i+1}}}{\partial\boldsymbol{S^{tri}_{i+1}}}
	\frac{\partial\boldsymbol{S^{tri}_{i+1}}}{\partial\boldsymbol{\sigma^{tri}_{i+1}}}:
	\frac{\partial\boldsymbol{\sigma^{tri}_{i+1}}}{\partial\boldsymbol{\epsilon^{etri}_{i+1}}}
\end{align}
$$
ここで
$$
\frac{\partial\boldsymbol{\sigma_{i+1}}}{\partial\boldsymbol{\sigma^{tri}_{i+1}}}=1
$$
$$
\frac{\partial\boldsymbol{\sigma^{tri}_{i+1}}}{\partial\boldsymbol{\epsilon^{etri}_{i+1}}}=
\boldsymbol{\tilde{D_e}}
$$
$$
\frac{\partial\boldsymbol{\sigma_{i+1}}}{\Delta\gamma}=-\sqrt{6}G\boldsymbol{\bar N^{i+1}_\sigma}
$$
$$
\begin{align}
\frac{\partial\boldsymbol{\sigma_{i+1}}}{\partial\boldsymbol{S^{tri}_{i+1}}} &=
- \frac{\sqrt{6}G\Delta\gamma}{||\boldsymbol{S^{tri}_{i+1}}||} \left(
	 \boldsymbol{\tilde{I}} -
	\frac{\boldsymbol{S^{tri}_{i+1}}}{||\boldsymbol{S^{tri}_{i+1}}||} \otimes \boldsymbol{\bar N^{i+1}_\sigma} 
\right) \\
&= - \frac{3G\Delta\gamma}{q^{tri}_{i+1}} \left(
	\boldsymbol{\tilde{I}} -
	\boldsymbol{\bar N^{i+1}_\sigma} \otimes \boldsymbol{\bar N^{i+1}_\sigma} 
\right) \\
\end{align}
$$
$$
\frac{\partial\boldsymbol{S^{tri}_{i+1}}}{\partial\boldsymbol{\sigma^{tri}_{i+1}}}=\boldsymbol{\tilde{I_d}}
$$

また
$$
\frac{\partial\Delta\gamma}{\partial\boldsymbol{\epsilon^{etri}_{i+1}}}=-\frac{\frac{\partial F}{\boldsymbol{\epsilon^{etri}_{i+1}}}}
{\frac{\partial F}{\partial\Delta\gamma}}=-\frac{\frac{\partial F}{\boldsymbol{\epsilon^{etri}_{i+1}}}}
{F'(\Delta\gamma)}
$$

今
$$
\begin{align}
F &= \sigma_{y0} + \theta\left(R_i + bQ\Delta\gamma\right)+
3G\Delta\gamma-q^{tri}_{i+1} \\
&= F\left(
	\boldsymbol{S^{tri}_{i+1}}\left(
		\boldsymbol{\sigma^{tri}_{i+1}}(\boldsymbol{\epsilon^{etri}_{i+1}})
	\right)
\right)
\end{align}
$$

$$
\begin{align}
\frac{\partial F}{\boldsymbol{\epsilon^{etri}_{i+1}}} &=
\frac{\partial F}{\partial \boldsymbol{S^{tri}_{i+1}}}
\frac{\partial \boldsymbol{S^{tri}_{i+1}}}{\partial\boldsymbol{\sigma^{tri}_{i+1}}}
\frac{\partial \boldsymbol{\sigma^{tri}_{i+1}}}{\partial\boldsymbol{\epsilon^{etri}_{i+1}}} \\
&= -\sqrt{\frac{3}{2}} \boldsymbol{\bar N^{i+1}_\sigma} \cdot \boldsymbol{\tilde{I_d}}:\boldsymbol{\tilde{D_e}} = -\sqrt{6}G\boldsymbol{\bar N^{i+1}_\sigma}
\end{align}
$$
よって
$$
\frac{\partial\Delta\gamma}{\partial\boldsymbol{\epsilon^{etri}_{i+1}}}=\frac{\sqrt{6}G{}}{F'(\Delta\gamma)}\boldsymbol{\bar N^{i+1}_\sigma}
$$

以上から$\boldsymbol{\tilde{D_{ep}}}$は下記となる.
$$
\begin{align}
\boldsymbol{\tilde{D_{ep}}} &=
\boldsymbol{\tilde{D_e}}-
\frac{6G^2}{F'(\Delta\gamma)}
\boldsymbol{\bar N^{i+1}_\sigma} \otimes \boldsymbol{\bar N^{i+1}_\sigma}   -
 \frac{3G\Delta\gamma }{q^{tri}_{i+1}} \left(
	\boldsymbol{\tilde{I}} -
	\boldsymbol{\bar N^{i+1}_\sigma} \otimes \boldsymbol{\bar N^{i+1}_\sigma}
\right) \cdot \boldsymbol{\tilde{I_d}}:\boldsymbol{\tilde{D_e}} \\
&= \boldsymbol{\tilde{D_e}}-
\frac{6G^2}{F'(\Delta\gamma)}
\boldsymbol{\bar N^{i+1}_\sigma} \otimes \boldsymbol{\bar N^{i+1}_\sigma}   -
 \frac{6G^2\Delta\gamma}{q^{tri}_{i+1}} \left(
	\boldsymbol{{\tilde{I_d}}} -
	\boldsymbol{\bar N^{i+1}_\sigma} \otimes \boldsymbol{\bar N^{i+1}_\sigma} 
\right) \\
&= \boldsymbol{\tilde{D_e}} - \frac{6G^2\Delta\gamma}{q^{tri}_{i+1}}\boldsymbol{\tilde{I_d}} +
6G^2 \left(
	\frac{\Delta\gamma}{q^{tri}_{i+1}} - \frac{1}{F'(\Delta\gamma)}
\right) \boldsymbol{\bar N^{i+1}_\sigma} \otimes \boldsymbol{\bar N^{i+1}_\sigma}
\end{align}
$$

