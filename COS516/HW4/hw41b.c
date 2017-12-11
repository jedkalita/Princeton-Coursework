int buf[4];
int hi = 0;
int lo = 0;
int size = 4;

int nondet_int() { int x; return x; }

/** FUNCTION enqueue */
void enqueue (int x)
{
    buf [hi] = x;
    hi = (hi + 1) % size;
}

/** FUNCTION dequeue */
int dequeue ()
{
    int res = buf [lo];
    lo = (lo + 1) % size;
    return res;
}

/** FUNCTION queue_test */
void queue_test (void)
{
    while (nondet_int())
    {
        if (nondet_int ())
        {
            int x = nondet_int ();
            enqueue (x);
        }
        else
            dequeue ();
    }
}



/** FUNCTION binsearch */
int binsearch (int x)
{
    int a[16];
    signed low = 0, high = 16;
    
    while (low < high)
    {
        signed middle = low + ((high - low) >> 1);
        
        if (a[middle]<x)
            high = middle;
        else if (a [middle] > x)
            low = middle + 1;
        else /* a [middle] == x ! */
            return middle;
    }
    return -1;
}
