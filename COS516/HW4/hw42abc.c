#include <assert.h>
/** prototypes */
int l = 0;
int nondet_int ();
void lock (void) {l++; };
void unlock (void) {l--;};
//int l = 0;

int read_input (void) { return nondet_int (); }

/** FUNCTION lock1 */
void lock1 (void)
{
    int in_irq;
    int buf[10];
    int i;
    
    if (in_irq)
    {
        //l++;
        lock ();
        //l++;
        assert(l != 2);
    }
    
    for (i = 0; i < 5; i++)
        buf [i] = read_input ();
    
    if (in_irq)
    {
        //l++;
        lock ();
        //l++;
        assert(l != 2);
    }
}

/** FUNCTION lock2 */
void lock2 (void)
{
    int request, old, total;
    
    request = 0;
    do {
        lock ();
        assert(l != 2);
        old = total;
        request = read_input ();
        if (request)
        {
            unlock ();
            assert(l != -1);
            total = total + 1;
        }
    } while (total != old);
    //assert (l == 1 && total == old);
    unlock ();
    assert(l != -1);
}
