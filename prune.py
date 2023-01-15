def prune(
        checkpoint,
        fp16 = False,
        ema = False,
        clip = True,
        vae = True,
        depth = True,
        unet = True,
):
    sd = checkpoint
    sd_pruned = dict()
    for k in sd:
        cp = unet and k.startswith('model.diffusion_model.')
        cp = cp or (depth and k.startswith('depth_model.'))
        cp = cp or (vae and k.startswith('first_stage_model.'))
        cp = cp or (clip and k.startswith('cond_stage_model.'))
        if cp:
            k_in = k
            if ema:
                k_ema = 'model_ema.' + k[6:].replace('.', '')
                if k_ema in sd:
                    k_in = k_ema
            sd_pruned[k] = sd[k_in].half() if fp16 else sd[k_in]
    return { 'state_dict': sd_pruned }


def main(args):
    from argparse import ArgumentParser
    from functools import partial
    parser = ArgumentParser(
        description="Prune a stable diffusion checkpoint"
    )
    parser.add_argument(
        'input',
        type=str,
        help="input checkpoint",
        nargs='?'
    )
    parser.add_argument(
        'output',
        type=str,
        help="output checkpoint",
        nargs='?'
    )
    parser.add_argument(
        '-s', '--safe',
        action='store_true',
        help="convert to safetensors when using batch"
    )
    parser.add_argument(
        '-p', '--fp16',
        action='store_true',
        help="convert to float16"
    )
    parser.add_argument(
        '-e', '--ema',
        action='store_true',
        help="use EMA for weights"
    )
    parser.add_argument(
        '-c', '--no-clip',
        action='store_true',
        help="strip CLIP weights"
    )
    parser.add_argument(
        '-a', '--no-vae',
        action='store_true',
        help="strip VAE weights"
    )
    parser.add_argument(
        '-d', '--no-depth',
        action='store_true',
        help="strip depth model weights"
    )
    parser.add_argument(
        '-u', '--no-unet',
        action='store_true',
        help="strip UNet weights"
    )
    parser.add_argument(
        '-o', '--overwrite',
        action='store_true',
        help="overwrite original file"
    )
    parser.add_argument(
        '-b', '--batch',
        action='store_true',
        help="pruned all models in folder"
    )
    


    def error(self, message):
        import sys
        sys.stderr.write(f"error: {message}\n")
        self.print_help()
        self.exit()
    parser.error = partial(error, parser)  # type: ignore
    args = parser.parse_args(args)

    if args.input == None and args.batch == False:
        raise ValueError("You need to provide input file if not using the -b --batch argument")
        
    if args.input != None and args.batch == False and args.overwrite == False and args.output == None:
        raise ValueError("You need to provide output file if not using the -o --overwrite argument")
        
    class torch_pickle:
        import pickle as python_pickle

        class Unpickler(python_pickle.Unpickler):
            def find_class(self, module, name):
                try:
                    return super().find_class(module, name)
                except:
                    return None
                    
    from safetensors.torch import load_file, save_file
    from torch import save, load
    from pickle import UnpicklingError
    
    if args.batch:
        import os
        files = os.listdir('.')
        
        for file in files:
            try:
                if file.endswith('.ckpt'):
                    try:
                        data = load(file, pickle_module = torch_pickle)
                    except UnpicklingError:
                        continue
                    if args.safe:
                        print("file : " + file)
                        print("file2 : " + file)
                        print("file3 : " + file)
                        save_file(prune(
                            load(file, pickle_module = torch_pickle)['state_dict'] if 'state_dict' in data else load(file, pickle_module = torch_pickle),
                            fp16=args.fp16,
                            ema=args.ema,
                            clip=not args.no_clip,
                            vae=not args.no_vae,
                            depth=not args.no_depth,
                            unet=not args.no_unet
                        )["state_dict"],file.rstrip('.ckpt') + '.safetensors' if args.overwrite else file.rstrip('.ckpt') + '-pruned.safetensors')
                    else:
                        save(prune(
                            load(file, pickle_module = torch_pickle)['state_dict'] if 'state_dict' in data else load(file, pickle_module = torch_pickle),
                            fp16 = args.fp16,
                            ema = args.ema,
                            clip = not args.no_clip,
                            vae = not args.no_vae,
                            depth = not args.no_depth,
                            unet = not args.no_unet
                        ), file if args.overwrite else file.rstrip('.ckpt') + '-pruned.ckpt')
                elif file.endswith('.safetensors'):  
                    save_file(prune(
                        load_file(file),
                        fp16=args.fp16,
                        ema=args.ema,
                        clip=not args.no_clip,
                        vae=not args.no_vae,
                        depth=not args.no_depth,
                        unet=not args.no_unet
                    )["state_dict"], file if args.overwrite else file.rstrip('.ckpt') + '-pruned.safetensors')
            except:
                print("error when pruning : {}".format(file))
                continue
    else:
        if args.input.endswith('.ckpt'):
            try:
                data = load(args.input, pickle_module = torch_pickle)
            except:
                raise ValueError("error, the code is broken :momijiwide: {}".format(args.input))
                
            pruned = prune(
                load(args.input, pickle_module = torch_pickle)['state_dict'] if 'state_dict' in data else load(args.input, pickle_module = torch_pickle),
                fp16=args.fp16,
                ema=args.ema,
                clip=not args.no_clip,
                vae=not args.no_vae,
                depth=not args.no_depth,
                unet=not args.no_unet
            )
            save(pruned, args.input if args.overwrite else args.output)
        elif args.input.endswith('.safetensors'):
            save_file(prune(
                load_file(args.input),
                fp16=args.fp16,
                ema=args.ema,
                clip=not args.no_clip,
                vae=not args.no_vae,
                depth=not args.no_depth,
                unet=not args.no_unet
            )["state_dict"], args.input if args.overwrite else args.output)
        else:
            raise ValueError("Unrecognized file extension, provide .ckpt or .safetensors file in arguments: {}".format(args.input))
            

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
