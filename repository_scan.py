import sys

import pandas as pd
from pydriller import Repository
from pydriller.metrics.process.change_set import ChangeSet
from pydriller.metrics.process.code_churn import CodeChurn
from pydriller.metrics.process.commits_count import CommitsCount
from pydriller.metrics.process.contributors_count import ContributorsCount
from pydriller.metrics.process.contributors_experience import ContributorsExperience
from pydriller.metrics.process.hunks_count import HunksCount
from pydriller.metrics.process.lines_count import LinesCount


def main():
    cve = pd.read_csv("../data/bigvul.csv")

    for i in range(len(cve)):
        commit_link = cve.iloc[i]['ref_link']
        vulnerable_hash = cve.iloc[i]['version_before_fix']
        fixed_hash = cve.iloc[i]['version_after_fix']
        print(commit_link)
        print(vulnerable_hash)
        print(fixed_hash)

        # Removal of user and commit hash from link
        repo_link = commit_link[:int(len(commit_link) - 48)]
        print(repo_link)

        # for commit in Repository(repo_link, single=vulnerable_hash, clone_repo_to="C:/Users/Matteo Cicalese/Downloads/dir_temp").traverse_commits():
        #     print(commit.dmm_unit_size)
        #     print(commit.dmm_unit_complexity)
        #     print(commit.dmm_unit_interfacing)
        #
        # for commit in Repository(repo_link, single=fixed_hash, clone_repo_to="C:/Users/Matteo Cicalese/Downloads/dir_temp").traverse_commits():
        #     print(commit.dmm_unit_size)
        #     print(commit.dmm_unit_complexity)
        #     print(commit.dmm_unit_interfacing)

        try:
            change_set = ChangeSet(path_to_repo=repo_link,
                                   from_commit=vulnerable_hash,
                                   to_commit=fixed_hash)
            maximum = change_set.max()
            average = change_set.avg()
            print('Maximum number of files committed together: {}'.format(maximum))
            print('Average number of files committed together: {}'.format(average))
        except Exception as e:
            print("Questa repo é stefanizzata")

        # code_churn = CodeChurn(path_to_repo='path/to/the/repo',
        #                        from_commit='from commit hash',
        #                        to_commit='to commit hash')
        # files_count = code_churn.count()
        # files_max = code_churn.max()
        # files_avg = code_churn.avg()
        # print('Total code churn for each file: {}'.format(files_count))
        # print('Maximum code churn for each file: {}'.format(files_max))
        # print('Average code churn for each file: {}'.format(files_avg))
        #
        # commits_count = CommitsCount(path_to_repo='path/to/the/repo',
        #                              from_commit='from commit hash',
        #                              to_commit='to commit hash')
        # files = commits_count.count()
        # print('Files: {}'.format(files))
        #
        # contributors_count = ContributorsCount(path_to_repo='path/to/the/repo',
        #                                        from_commit='from commit hash',
        #                                        to_commit='to commit hash')
        # count = contributors_count.count()
        # minor = contributors_count.count_minor()
        # print('Number of contributors per file: {}'.format(count))
        # print('Number of "minor" contributors per file: {}'.format(minor))
        #
        # contributors_experience = ContributorsExperience(path_to_repo='path/to/the/repo',
        #                                                  from_commit='from commit hash',
        #                                                  to_commit='to commit hash')
        # files = contributors_experience.count()
        # print('Files: {}'.format(files))
        #
        # hunks_count = HunksCount(path_to_repo='path/to/the/repo',
        #                          from_commit='from commit hash',
        #                          to_commit='to commit hash')
        # files = hunks_count.count()
        # print('Files: {}'.format(files))
        #
        # lines_count = LinesCount(path_to_repo='path/to/the/repo',
        #                          from_commit='from commit hash',
        #                          to_commit='to commit hash')
        #
        # added_count = lines_count.count_added()
        # added_max = lines_count.max_added()
        # added_avg = lines_count.avg_added()
        # print('Total lines added per file: {}'.format(added_count))
        # print('Maximum lines added per file: {}'.format(added_max))
        # print('Average lines added per file: {}'.format(added_avg))
        #
        # removed_count = lines_count.count_removed()
        # removed_max = lines_count.max_removed()
        # removed_avg = lines_count.avg_removed()
        # print('Total lines removed per file: {}'.format(removed_count))
        # print('Maximum lines removed per file: {}'.format(removed_max))
        # print('Average lines removed per file: {}'.format(removed_avg))

        print("\n\n\n\n\nEND OF REPOSITORY COMMITS\n\n\n\n\n")


if __name__ == "__main__":
    main()
