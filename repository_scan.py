import csv
import shutil
import sys

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
from github import Github


def metric_calculation_and_writing(start, to, commit_link, writer, label):
    print("Calcolo metriche...")
    code_churn = CodeChurn(path_to_repo=commit_link,
                           from_commit=start,
                           to_commit=to)
    files_count = code_churn.count()
    files_max = code_churn.max()
    files_avg = code_churn.avg()
    filename = files_count.keys()

    commits_count = CommitsCount(path_to_repo=commit_link,
                                 from_commit=start,
                                 to_commit=to)
    numCommit = commits_count.count()

    contributors_count = ContributorsCount(path_to_repo=commit_link,
                                           from_commit=start,
                                           to_commit=to)
    count = contributors_count.count()
    minor = contributors_count.count_minor()

    contributors_experience = ContributorsExperience(path_to_repo=commit_link,
                                                     from_commit=start,
                                                     to_commit=to)
    contrExp = contributors_experience.count()

    hunks_count = HunksCount(path_to_repo=commit_link,
                             from_commit=start,
                             to_commit=to)
    numHunks = hunks_count.count()

    lines_count = LinesCount(path_to_repo=commit_link,
                             from_commit=start,
                             to_commit=to)

    added_count = lines_count.count_added()
    added_max = lines_count.max_added()
    added_avg = lines_count.avg_added()

    removed_count = lines_count.count_removed()
    removed_max = lines_count.max_removed()
    removed_avg = lines_count.avg_removed()

    for file in filename:
        writer.writerow([file, files_count.get(file), files_max.get(file), files_avg.get(file), numCommit.get(file),
                         count.get(file), minor.get(file), contrExp.get(file), numHunks.get(file),
                         added_count.get(file), added_max.get(file), added_avg.get(file), removed_count.get(file),
                         removed_max.get(file), removed_avg.get(file), label])


def get_commit_count(repo_link):

    parts = repo_link.split("/")
    username = parts[-2]
    repository = parts[-1]

    g = Github()
    repo = g.get_repo(f"{username}/{repository}")
    commit_count = repo.get_commits().totalCount

    return commit_count


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
            try:
                commit_link = cve.iloc[i]['repository']
                fixed_hash = cve.iloc[i]['fixed_hash']
                vulnerable_hash = None
                pre_fix_hash = None
                last_hash = None

                print("\n\nRepo link: ", commit_link)

                # for commit in Repository(repo_link, single=vulnerable_hash).traverse_commits():
                #     print(commit.dmm_unit_size)
                #     print(commit.dmm_unit_complexity)
                #     print(commit.dmm_unit_interfacing)
                #
                # for commit in Repository(repo_link, single=fixed_hash).traverse_commits():
                #     print(commit.dmm_unit_size)
                #     print(commit.dmm_unit_complexity)
                #     print(commit.dmm_unit_interfacing)

                num_commit = get_commit_count(commit_link)
                if num_commit > 40000:
                    print("\nSkippata repo con numero di commit: ", num_commit)
                    continue

                repository = Repository(commit_link, single=fixed_hash)
                for commit in repository.traverse_commits():
                    # commit vulnerabile
                    print("Trovo il commit vulnerabile più obsoleto")
                    buggy_commits = repository.git.get_commits_last_modified_lines(commit)

                    for key, value in buggy_commits.items():
                        vulnerable_commit = repository.git.get_commit(value.pop())
                        for i in range(len(value)):
                            commit_new = repository.git.get_commit(value.pop())
                            data_older = vulnerable_commit.committer_date
                            data_new = commit_new.committer_date
                            if data_new < data_older:
                                vulnerable_commit = commit_new
                        vulnerable_hash = vulnerable_commit.hash

                    # commit precedente al commit fixato: converto il generator di tutti i commit in un oggetto list,
                    # poi prendo l'indice del commit fixato all'interno della lista in modo da ottenere il commit
                    # precedente al fixato
                    commit_generator = repository.git.get_list_commits()
                    array_commit = list(commit_generator)
                    index_commit = array_commit.index(repository.git.get_commit(fixed_hash))
                    pre_fix = array_commit[index_commit - 1]
                    pre_fix_hash = pre_fix.hash

                    # ultimo commit
                    last_commit = repository.git.get_head()
                    last_hash = last_commit.hash

                    if vulnerable_hash is None or last_hash is None or pre_fix_hash is None:
                        print("\nErrore nella ricerca dei commit \n")
                        continue

                print("Fixed hash: ", fixed_hash)
                print("Pre fix hash: ", pre_fix_hash)
                print("Vulnerable hash: ", vulnerable_hash)
                print("Last hash: ", last_hash)

                metric_calculation_and_writing(vulnerable_hash, pre_fix_hash, commit_link, writer, 1)
                metric_calculation_and_writing(fixed_hash, last_hash, commit_link, writer, 0)
            except Exception as e:
                sys.stderr.write(str(e))
                continue


if __name__ == "__main__":
    main()
