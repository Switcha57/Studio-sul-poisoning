from libs.find_adv import find_adv

step_range = [1000, 100, 10]


def adv_step_mode(mode, differences_index, target, index_file, model, id_file, STEP_MATCH, folder, section_n, dir_pe,
                  resume, OPTIMIZED,c,injection_type):

    if mode == "OneShot":
        res = find_adv(differences_index, target, index_file, model, id_file, STEP_MATCH, folder, section_n, dir_pe,
                       resume, OPTIMIZED,c,injection_type)
        return res
    elif mode == "IterOnSection":

        for i_section in list(reversed(range(1, section_n+1))):
            res = find_adv(differences_index, target, index_file, model, id_file, STEP_MATCH, folder, i_section,
                           dir_pe,
                           resume, OPTIMIZED,c,injection_type)
            if res["res"] is not None:
                return res
        return False

    elif mode == "IterOnStep":
        for i_step in step_range:
            res = find_adv(differences_index, target, index_file, model, id_file, i_step, folder, section_n,
                           dir_pe,
                           resume, OPTIMIZED,c,injection_type)
            if res["res"] is not None:
                return res
        return False

    elif mode == "all":
        for i_step in step_range:
            for i_section in list(reversed(range(1, section_n + 1))):
                res = find_adv(differences_index, target, index_file, model, id_file, i_step, folder, i_section,
                               dir_pe,
                               resume, OPTIMIZED,c,injection_type)
                if res["res"] is not None:
                    return res
