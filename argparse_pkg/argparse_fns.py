import os


class VideoNotPassedError(Exception):
    pass

class VideoDoesNotExistError(Exception):
    pass

class TargetFolderDoesNotExist(Exception):
    pass

class TargetFolderNotPassed(Exception):
    pass


def getopts(argv):
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts

def parse_arguments(args):

    parse_dict = {}

    if('-v' in args.keys()):
        video = args['-v']
        if(os.path.exists(video)):
            parse_dict['video']=video
        else:
            raise VideoDoesNotExistError("Passed video does not exist")
    else:
        raise VideoNotPassedError("Pass video as an argument with -v")

    if('-t' in  args.keys()):
        target_folder = args['-t']

        if(os.path.isdir(target_folder)):
            parse_dict['target_folder']=target_folder
        else:
            raise TargetFolderDoesNotExist("Target Folder does not exist")

    else:
        raise TargetFolderNotPassed("Target Folder must be passed with a -t")

    if ('-debug' in args.keys()):
        debug = bool(args['-debug'])
        parse_dict['debug'] = debug

    else:
        parse_dict['debug'] = False


    return parse_dict