

def model_comparison_heatmap(df_results):

    df_results.sort_values(['avg_test_score'], ascending=False).head(10)

    models = np.unique(df_results['model'])
    selectors = np.unique(df_results['selector'])

    heatm_data = np.zeros((models.size, selectors.size), dtype=float)
    std_labels = np.zeros((models.size, selectors.size), dtype=float)

    model_grps = df_results.groupby('model')
    for mod_num, (name, model_grp) in enumerate(model_grps):
        #print(name)
        model_sel_grps = model_grp.groupby('selector')
        for sel_num, (name, model_sel_grp) in enumerate(model_sel_grps):
            heatm_data[mod_num, sel_num] = np.round(
                np.mean(model_sel_grp['avg_test_score']), decimals=2
            )
            std_labels[mod_num, sel_num] = np.round(
                np.std(model_sel_grp['avg_test_score']), decimals=2
            )

    plt.figure(figsize=(10,10))
    ax = sns.heatmap(
        heatm_data, annot=True, fmt = '.2f', square=1, linewidth=.5, cbar=True,
        cmap='viridis', cbar_kws={'shrink': 0.6}, annot_kws={'size': 10}
    )
    for num, value in enumerate(ax.texts):
        value.set_text('{} +/- {}'.format(value.get_text(), std_labels.ravel()[num]))

    plt.yticks(np.arange(models.size), models, fontsize=10, rotation=45)
    plt.xticks(np.arange(selectors.size), selectors, fontsize=10, rotation=45)
    plt.tight_layout()


if __name__ == '__main__':

    results = pd.read_csv('./../data/results/model_comparison/model_comparison_results.csv')
