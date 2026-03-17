from methods import register


@register('test')
def method(args):
    print(args)
