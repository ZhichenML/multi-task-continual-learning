import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
NUM_STYLES = len(LINE_STYLES)



def print_eval_matrix(eval_matrix):
    for i in range(len(eval_matrix)):
        print(eval_matrix[i])
        print("\n")

def plot(path="./outputs/cner_output/bert/span_eval_matrix.npz"):
    res = np.load(path, allow_pickle=True)
    # eval_matrix=res["eval_matrix"] 
    eval_matrix=res["eval_matrix_cates"]
    num_task = len(eval_matrix)

    from collections import defaultdict
    ner_result = defaultdict(list)
    cate_result = defaultdict(list)
    
    # 取出每个task的metric score dict
    for i in range(num_task):
        ner_result[i] = [eval_matrix[j][i] for j in range(i, num_task)]
    
    # 从每个task的score dict中取出要plot的score
    score_name = ["acc", "recall", "f1", "loss"]
    # score_name = ["acc"]
    for name in score_name:
        # plt.figure()
        # ax=plt.subplot(1,1,1)
        # # ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(30)])
        # cm = plt.get_cmap('gist_rainbow')
        # ax.set_prop_cycle(color=[cm(1.*i/30) for i in range(30)])
        sns.reset_orig()  # get default matplotlib styles back
        clrs = sns.color_palette('husl', n_colors=num_task)  # a list of RGB tuples
        fig, ax = plt.subplots(1)
    
            
        cl_res = []
        task_name=[]
        for i in range(num_task):    
            cl_res.append([v[name]  for v in ner_result[i]])
            task_name.append(ner_result[i][0]['task_name'])
        
        for i in range(num_task):
            for j in range(i):
                cl_res[i].insert(0,None)
        
        for i in range(num_task):
            # plt.plot(cl_res[i], label="task %s" %i)
            lines = ax.plot(cl_res[i], label="task %s" %task_name[i])
            lines[0].set_color(clrs[i])
            lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])


        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                        box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                fancybox=True, shadow=True, ncol=5)

        plt.xlabel("Num of Tasks")
        plt.ylabel(name)
        plt.xlim(0,num_task)
        plt.show()
        savefig_path = "./outputs/Fig/"+ name+"_cl.jpg"
        plt.savefig(savefig_path, bbox_inches='tight', dpi=300)
    

if __name__ == "__main__":
    plot()
