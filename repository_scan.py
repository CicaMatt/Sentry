import csv
import shutil
import git
import pandas as pd
from pydriller import Repository, Git
from pydriller.metrics.process.change_set import ChangeSet
from pydriller.metrics.process.code_churn import CodeChurn
from pydriller.metrics.process.commits_count import CommitsCount
from pydriller.metrics.process.contributors_count import ContributorsCount
from pydriller.metrics.process.contributors_experience import ContributorsExperience
from pydriller.metrics.process.hunks_count import HunksCount
from pydriller.metrics.process.lines_count import LinesCount


def metric_calculation_and_writing(start, to, commit_link, writer, label):

    code_churn = CodeChurn(path_to_repo=commit_link,
                           from_commit=start,
                           to_commit=to)
    files_count = code_churn.count()
    files_max = code_churn.max()
    files_avg = code_churn.avg()
    print('Total code churn for each file: {}'.format(files_count), files_count.keys())
    print('Maximum code churn for each file: {}'.format(files_max))
    print('Average code churn for each file: {}'.format(files_avg))

    filename = files_count.keys()

    commits_count = CommitsCount(path_to_repo=commit_link,
                                 from_commit=start,
                                 to_commit=to)
    numCommit = commits_count.count()
    print('Files: {}'.format(numCommit))

    contributors_count = ContributorsCount(path_to_repo=commit_link,
                                           from_commit=start,
                                           to_commit=to)
    count = contributors_count.count()
    minor = contributors_count.count_minor()
    print('Number of contributors per file: {}'.format(count))
    print('Number of "minor" contributors per file: {}'.format(minor))

    contributors_experience = ContributorsExperience(path_to_repo=commit_link,
                                                     from_commit=start,
                                                     to_commit=to)
    contrExp = contributors_experience.count()
    print('contrExp: {}'.format(contrExp))

    hunks_count = HunksCount(path_to_repo=commit_link,
                             from_commit=start,
                             to_commit=to)
    numHunks = hunks_count.count()
    print('Num Hunks: {}'.format(numHunks))

    lines_count = LinesCount(path_to_repo=commit_link,
                             from_commit=start,
                             to_commit=to)

    added_count = lines_count.count_added()
    added_max = lines_count.max_added()
    added_avg = lines_count.avg_added()
    print('Total lines added per file: {}'.format(added_count))
    print('Maximum lines added per file: {}'.format(added_max))
    print('Average lines added per file: {}'.format(added_avg))

    removed_count = lines_count.count_removed()
    removed_max = lines_count.max_removed()
    removed_avg = lines_count.avg_removed()
    print('Total lines removed per file: {}'.format(removed_count))
    print('Maximum lines removed per file: {}'.format(removed_max))
    print('Average lines removed per file: {}'.format(removed_avg))

    for file in filename:
        writer.writerow([file, files_count.get(file), files_max.get(file), files_avg.get(file), numCommit.get(file),
                         count.get(file), minor.get(file), contrExp.get(file), numHunks.get(file),
                         added_count.get(file), added_max.get(file), added_avg.get(file), removed_count.get(file),
                         removed_max.get(file), removed_avg.get(file), label])


def main():
    cve = pd.read_csv("./data/CVEfixes.csv")
    filename = 'dataset.csv'

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['', 'filename', 'MaxNumCommittedFiles', 'AvgNumCommittedFiles', "#CodeChurnInFile",
                         "MaxSizeCodeChurn", "AvgSizeCodeChurn", "#Commits", "#Contributors",
                         "#MinorContributors", "#ContibutorExperience", "#Hunks", "#LinesAdded", "MaxLinesAdded",
                         "AvgLinesAdded", "#LinesRemoved", "MaxLinesRemoved", "AvgLinesRemoved", "vulnerable"])

        for i in range(len(cve)):
            stop = 0
            try:
                commit_link = cve.iloc[i]['repository']
                fixed_hash = cve.iloc[i]['fixed_hash']
                vulnerable_hash = None
                pre_fix_hash = None
                last_hash = None

                # for commit in Repository(repo_link, single=vulnerable_hash).traverse_commits():
                #     print(commit.dmm_unit_size)
                #     print(commit.dmm_unit_complexity)
                #     print(commit.dmm_unit_interfacing)
                #
                # for commit in Repository(repo_link, single=fixed_hash).traverse_commits():
                #     print(commit.dmm_unit_size)
                #     print(commit.dmm_unit_complexity)
                #     print(commit.dmm_unit_interfacing)

                repository = Repository(commit_link, single=fixed_hash)
                for commit in repository.traverse_commits():
                    if repository.git.total_commits() > 100000:
                        stop = 1
                        break
                    # commit vulnerabile
                    buggy_commits = repository.git.get_commits_last_modified_lines(commit)
                    print(buggy_commits)
                    for key, value in buggy_commits.items():
                        vulnerable_commit = repository.git.get_commit(value.pop())
                        for i in range(len(value)):
                            commit_new = repository.git.get_commit(value.pop())
                            data_older = vulnerable_commit.committer_date
                            data_new = commit_new.committer_date
                            if data_new < data_older:
                                vulnerable_commit = commit_new
                        vulnerable_hash = vulnerable_commit.hash

                    # commit precedente al commit fixato: converto il generator di tutti i commit in un oggetto list, poi prendo
                    # l'indice del commit fixato all'interno della lista in modo da ottenere il commit precedente al fixato
                    commit_generator = repository.git.get_list_commits()
                    array_commit = list(commit_generator)
                    index_commit = array_commit.index(repository.git.get_commit(fixed_hash))
                    pre_fix = array_commit[index_commit - 1]
                    pre_fix_hash = pre_fix.hash

                    # ultimo commit
                    last_commit = repository.git.get_head()
                    last_hash = last_commit.hash

                if stop:
                    print("Skippata repo con numero di commit: ", repository.git.total_commits())
                    continue
                print("Fixed hash: ", fixed_hash)
                print("Pre fix hash: ", pre_fix_hash)
                print("Vulnerable hash: ", vulnerable_hash)
                print("Last hash: ", last_hash)

                metric_calculation_and_writing(vulnerable_hash, pre_fix_hash, commit_link, writer, 1)
                metric_calculation_and_writing(fixed_hash, last_hash, commit_link, writer, 0)
            except:
                continue


if __name__ == "__main__":
    main()