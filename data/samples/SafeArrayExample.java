public class SafeArrayExample {

    public int getLastElement(int[] numbers) {
        if (numbers == null || numbers.length == 0) {
            return -1;
        }

        return numbers[numbers.length - 1];
    }

}