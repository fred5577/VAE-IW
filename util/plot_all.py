import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()
sns.set_style("whitegrid")

data = pd.read_csv("new_normalized_results.csv")
data = data.sort_values(by=["Human"], ascending=False)
data['BPROST'] = data['B-PROST']
data['Domain'] = [d.strip() for d in data['Domain']]
data = data.drop(columns=['B-PROST', 'col'])

lim1, lim2 = 1300, 19900


def plot_adjust():
    # plt.gca().set_facecolor('white')
    plt.xlabel(None)
    plt.ylabel(None)
    ax.tick_params(axis=u'y', which=u'both', length=0)
    sns.despine(left=True, bottom=False, trim=True)
    # plt.grid(axis='x')
    plt.tight_layout()


def mask(df, cond):
    out = df.copy()
    out.loc[cond, 'Human'] = 0
    out.loc[cond, 'BPROST'] = 0
    return out


# Make a dataframe where all entries such that VAE>BPROST are 0. For these
# we will plot BPROST first, then VAE.
cond = data.Human >= data.BPROST
data1 = mask(data, cond)

# Converse
cond = data.Human < data.BPROST
data2 = mask(data, cond)

# Plot settings
alpha = 1
p = sns.color_palette('deep')
color_ours = p[9]
color_bprost = (.75, .75, .75)
grey = sns.color_palette('dark')[7]

###############
# First plot

f, (ax) = plt.subplots(1, 1, figsize=(16, 13))
ax.set_xlim(0, lim1)
# ax.set_facecolor('white')

sns.barplot(x="BPROST", y="Domain", data=data1,
            label="B-PROST", color=color_bprost, alpha=alpha)
sns.barplot(x="Human", y="Domain", data=data1,
            label="VAE", color=color_ours, alpha=alpha)
sns.barplot(x="Human", y="Domain", data=data2,
            label="VAE", color=color_ours, alpha=alpha)
sns.barplot(x="BPROST", y="Domain", data=data2,
            label="B-PROST", color=color_bprost, alpha=alpha)

# plt.xticks([0, 200, 400, 600, 800, 1000])

ax.axhline(len(data) - 12.5, c=grey)
for i, v in enumerate(data.Human):
    txt = ax.text(12, i + 0.18, str(round(v)) + "%", color=grey, size=10)
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w',
                                                 alpha=0.7)])

plot_adjust()

f.savefig("1_wide_wide.pdf")

###############
# Second plot

f, (ax) = plt.subplots(1, 1, figsize=(6, 13))
ax.set_xlim(0, lim2)

sns.barplot(x="BPROST", y="Domain", data=data1,
            label="RA RolloutIW B-PROST", color=color_bprost, alpha=alpha)
sns.barplot(x="Human", y="Domain", data=data1,
            label="RA VAE-IW", color=color_ours, alpha=alpha)
sns.barplot(x="Human", y="Domain", data=data2,
            label=None, color=color_ours, alpha=alpha)
sns.barplot(x="BPROST", y="Domain", data=data2,
            label=None, color=color_bprost, alpha=alpha)

ax.axhline(len(data) - 12.5, c=grey)
ax.text(19900, len(data) - 12.5 - 0.1, "At human level or above", color=grey,
        ha='right', va='bottom', size=11)
ax.text(19800, len(data) - 12.5 + 0.15, "Below human level", color=grey,
        ha='right', va='top', size=11)

plt.legend(loc=4)

plt.xticks(range(0, 20000, 4000))
plot_adjust()

f.savefig("2_wide.pdf")
